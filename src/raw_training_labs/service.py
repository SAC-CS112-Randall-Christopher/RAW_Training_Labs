"""Shared operations for the CLI and local GUI; project authority lives here."""

import json
import time

from llm_lab.corpus import SPLITS, verify_corpus, write_corpus
from llm_lab.exposure import check as exposure_check
from llm_lab.io import canonical, digest, sha_file

from raw_training_labs.contracts import Import, Project, Review
from raw_training_labs.profiles import PROFILES, recipe_for, supported
from raw_training_labs.storage import Store, new_id


class Service:
    def __init__(self, root=None):
        self.store = Store() if root is None else Store(root)

    def create_project(self, payload):
        project = Project.model_validate(payload)
        project_id = new_id()
        with self.store.transaction() as db:
            if db.execute("SELECT count(*) FROM projects").fetchone()[0] >= 100:
                raise ValueError("Initial workbench limit is 100 projects")
            db.execute(
                "INSERT INTO projects VALUES(?,?,?)",
                (project_id, canonical(project.model_dump()), time.time()),
            )
        self.store.directory(project_id)
        return self.store.project(project_id)

    def projects(self):
        with self.store.connect() as db:
            rows = db.execute("SELECT id FROM projects ORDER BY created DESC LIMIT 100").fetchall()
            return [self.store.project(row["id"], db) for row in rows]

    def import_examples(self, project_id, payload):
        incoming = Import.model_validate(payload)
        self.store.project(project_id)
        identifiers = [e.id for e in incoming.examples]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Import contains duplicate example identifiers; nothing imported")
        with self.store.transaction() as db:
            if db.execute(
                "SELECT 1 FROM artifacts WHERE project=? AND kind='corpus'", (project_id,)
            ).fetchone():
                raise ValueError("Project is frozen; preserve it and create a new reviewed project")
            count = db.execute(
                "SELECT count(*) FROM examples WHERE project=?", (project_id,)
            ).fetchone()[0]
            if count + len(identifiers) > 64:
                raise ValueError(
                    "Initial bounded exposure/preparation supports at most 64 examples"
                )
            existing = {
                r[0] for r in db.execute("SELECT id FROM examples WHERE project=?", (project_id,))
            }
            if existing.intersection(identifiers):
                raise ValueError(
                    "Identifier already imported; preserve original and add a review revision"
                )
            db.executemany(
                "INSERT INTO examples VALUES(?,?,?)",
                [(project_id, e.id, canonical(e.model_dump())) for e in incoming.examples],
            )
        return {
            "imported": len(identifiers),
            "ids": identifiers,
            "review_status": "pending",
            "validation": self.validate(project_id),
        }

    def _examples(self, project_id, db, *, redact_test=False):
        self.store.project(project_id, db)
        rows = db.execute(
            """SELECT e.id,
            CASE WHEN ? AND json_extract(e.original,'$.split')='test'
              THEN json_set(e.original,'$.completion','[sealed final target]')
              ELSE e.original END original,
            CASE WHEN ? AND json_extract(e.original,'$.split')='test'
              THEN json_set(r.body,'$.completion','[sealed final target]')
              ELSE r.body END review,r.revision FROM examples e
            LEFT JOIN reviews r ON e.project=r.project AND e.id=r.example AND r.revision=
            (SELECT max(revision) FROM reviews WHERE project=e.project AND example=e.id)
            WHERE e.project=? ORDER BY e.id""",
            (redact_test, redact_test, project_id),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "original": json.loads(r["original"]),
                "review": json.loads(r["review"]) if r["review"] else None,
                "revision": r["revision"],
            }
            for r in rows
        ]

    def examples(self, project_id):
        with self.store.connect() as db:
            frozen = db.execute(
                "SELECT id FROM artifacts WHERE project=? AND kind='corpus'", (project_id,)
            ).fetchone()
            rows = self._examples(project_id, db, redact_test=bool(frozen))
        # Final targets are sealed once frozen. Original and reviewed teaching targets remain
        # visible for train/validation/regression; final targets appear only in final comparison.
        if frozen:
            for item in rows:
                if item["original"]["split"] == "test":
                    item["original"]["completion"] = "[sealed final target]"
                    if item["review"]:
                        item["review"]["completion"] = "[sealed final target]"
        return rows

    def review(self, project_id, example_id, payload):
        review = Review.model_validate(payload)
        with self.store.transaction() as db:
            if db.execute(
                "SELECT 1 FROM artifacts WHERE project=? AND kind='corpus'", (project_id,)
            ).fetchone():
                raise ValueError(
                    "Frozen examples cannot change; create a new independently reviewed project"
                )
            rows = self._examples(project_id, db)
            item = next((r for r in rows if r["id"] == example_id), None)
            if item is None:
                raise ValueError("Example not found in this project")
            if (
                review.reviewer_kind == "delegated_semantic"
                and item["original"]["source_kind"] != "raw_synthetic"
            ):
                raise ValueError(
                    "Delegated review admits RAW synthetic instructional material only"
                )
            if not review.rights.retention:
                raise ValueError("Retention permission is required")
            revision = (item["revision"] or 0) + 1
            db.execute(
                "INSERT INTO reviews VALUES(?,?,?,?,?)",
                (project_id, example_id, revision, canonical(review.model_dump()), time.time()),
            )
        return {"id": example_id, "revision": revision, "review": review.model_dump()}

    def _validate(self, project_id, rows, project):
        problems, prompts, groups, counts = [], {}, {}, dict.fromkeys(SPLITS, 0)
        chronological = {s: [] for s in SPLITS}
        for item in rows:
            original, review = item["original"], item["review"]
            eid, split = item["id"], original["split"]
            counts[split] += 1
            chronological[split].append(original["available_at"])
            key = digest(original["prompt"])
            target = review["completion"] if review else original["completion"]
            if key in prompts:
                old_id, old_split, old_target = prompts[key]
                code = "conflicting_target" if old_target != target else "duplicate_prompt"
                problems.append(
                    {
                        "id": eid,
                        "related_id": old_id,
                        "code": code,
                        "reason": "Repeated input; resolve explicitly before freezing",
                    }
                )
                if old_split != split:
                    problems.append({"id": eid, "code": "prompt_crosses_splits"})
            prompts[key] = (eid, split, target)
            group = original["group"]
            if group in groups and groups[group] != split:
                problems.append({"id": eid, "code": "group_crosses_splits"})
            groups[group] = split
            if not review or not review["approved"]:
                problems.append({"id": eid, "code": "review_required"})
                continue
            rights = review["rights"]
            if (
                not rights["retention"]
                or not rights["evaluation"]
                or (split == "train" and not rights["training"])
            ):
                problems.append({"id": eid, "code": "permission_missing"})
            if project["criteria"]["task"] == "work_request":
                try:
                    parsed = json.loads(target)
                    fields = project["criteria"]["fields"]
                    if not isinstance(parsed, dict) or set(parsed) != set(fields):
                        raise ValueError("Teaching target must contain exactly the declared fields")
                    if any(v is not None and not isinstance(v, str) for v in parsed.values()):
                        raise ValueError("Work request fields must be text or explicit null")
                except (ValueError, TypeError) as exc:
                    problems.append(
                        {"id": eid, "code": "invalid_teaching_target", "reason": str(exc)}
                    )
            if project["criteria"]["task"] == "decision" and not (
                len(target.strip()) == 1 and "A" <= target.strip() <= "X"
            ):
                problems.append({"id": eid, "code": "invalid_decision_target"})
        for split, count in counts.items():
            if not count:
                problems.append({"code": "empty_split", "split": split})
        for earlier, later in (("train", "validation"), ("validation", "test")):
            if (
                chronological[earlier]
                and chronological[later]
                and max(chronological[earlier]) >= min(chronological[later])
            ):
                problems.append({"code": "chronology_overlap", "splits": [earlier, later]})
        return {
            "eligible": not problems,
            "counts": counts,
            "problems": problems,
            "limitations": [
                "Explicit groups and exact inputs are checked; unrelated-looking "
                "near duplicates need semantic review."
            ],
        }

    def validate(self, project_id):
        with self.store.connect() as db:
            self.store.project(project_id, db)
            frozen = db.execute(
                "SELECT id FROM artifacts WHERE project=? AND kind='corpus'", (project_id,)
            ).fetchone()
            if frozen:
                artifact = self.store.artifact(project_id, frozen["id"], "corpus")
                manifest, _ = verify_corpus(artifact["path"], splits=())
                return {
                    "eligible": True,
                    "counts": artifact["body"]["counts"],
                    "problems": [],
                    "frozen": True,
                    "limitations": manifest["limitations"],
                }
            return self._validate(
                project_id, self._examples(project_id, db), self.store.project(project_id, db)
            )

    def prepare(self, project_id):
        self.store.budget(reserve=1024**2)
        with self.store.transaction() as db:
            if db.execute(
                "SELECT 1 FROM artifacts WHERE project=? AND kind='corpus'", (project_id,)
            ).fetchone():
                raise ValueError("Project already frozen; final targets cannot be recycled")
            rows = self._examples(project_id, db)
            project = self.store.project(project_id, db)
            validation = self._validate(project_id, rows, project)
            if not validation["eligible"]:
                raise ValueError("Preparation blocked: " + canonical(validation["problems"]))
            if db.execute(
                "SELECT 1 FROM artifacts WHERE project=? AND kind='corpus'", (project_id,)
            ).fetchone():
                raise ValueError(
                    "Project already frozen; do not recycle final targets into an iteration"
                )
            files, metadata, origins = {s: [] for s in SPLITS}, {}, {}
            for item in rows:
                original, review, eid = item["original"], item["review"], item["id"]
                split = original["split"]
                files[split].append(
                    {
                        "id": eid,
                        "role": "operator",
                        "prompt": original["prompt"],
                        "completion": [{"role": "assistant", "content": review["completion"]}],
                    }
                )
                metadata[eid] = {
                    "id": eid,
                    "split": split,
                    "families": [original["group"]],
                    "evidence_key": digest(original["prompt"]),
                    "data_basis": original["source_kind"],
                    "categories": [project["criteria"]["task"]],
                }
                origins[eid] = {
                    "original_sha256": digest(original),
                    "review_sha256": digest(review),
                    "review_revision": item["revision"],
                    "review": {k: v for k, v in review.items() if k != "completion"},
                    "source_kind": original["source_kind"],
                }
            previous = db.execute(
                "SELECT id,project,relative FROM artifacts WHERE kind='corpus'"
            ).fetchall()
            policy = {
                "studies": [
                    {
                        "id": r["id"],
                        "sealed": True,
                        "manifest": f"projects/{r['project']}/{r['relative']}/manifest.json",
                        "sha256": sha_file(
                            self.store.artifact_path(r["project"], r["relative"]) / "manifest.json"
                        ),
                    }
                    for r in previous
                ],
                "retirements": [],
            }
            protection = exposure_check(self.store.root, policy, list(metadata.values()))
            if not protection["eligible"]:
                raise ValueError(
                    "Protected exposure conflict with an existing RAW population; "
                    "related sealed examples cannot be reused as fresh data"
                )
            aid = new_id()
            relative = f"corpora/{aid}"
            destination = self.store.artifact_path(project_id, relative)
            destination.parent.mkdir(exist_ok=True)
            manifest = write_corpus(
                files,
                destination,
                {
                    "source": "raw-reviewed-neutral-v1",
                    "project_id": project_id,
                    "criteria": project["criteria"],
                    "criteria_sha256": digest(project["criteria"]),
                    "exposure_metadata": metadata,
                    "metadata": metadata,
                    "source_reviews": origins,
                    "protection": protection,
                    "regression_available": True,
                    "limitations": validation["limitations"]
                    + [
                        "Final targets sealed after initial review; "
                        "only baseline/final comparison jobs may expose them. "
                        "Synthetic evidence is not customer acceptance."
                    ],
                },
            )
            self.store.add_artifact(
                project_id,
                aid,
                "corpus",
                relative,
                {"sha256": manifest["corpus_sha256"], "counts": validation["counts"]},
                db,
            )
        return {"id": aid, "manifest": {k: v for k, v in manifest.items() if k != "source_reviews"}}

    def artifacts(self, project_id):
        self.store.project(project_id)
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT id,kind,body,created FROM artifacts WHERE project=? ORDER BY created",
                (project_id,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "kind": r["kind"],
                "body": json.loads(r["body"]),
                "created_at": r["created"],
            }
            for r in rows
        ]

    def preflight(self, project_id, corpus_id, profile_id):
        profile = supported(profile_id)
        artifact = self.store.artifact(project_id, corpus_id, "corpus")
        manifest, files = verify_corpus(artifact["path"], splits=("train", "validation"))
        if manifest["corpus_sha256"] != artifact["body"]["sha256"]:
            raise ValueError("Registered corpus identity changed")
        if any("completion" in p["review"] for p in manifest["source_reviews"].values()):
            raise ValueError(
                "Legacy draft corpus contains targets in review metadata; "
                "it is retained as provenance but not admitted for model jobs"
            )
        base = self.store.artifact_path(project_id, f"models/{profile_id}")
        if not (base / "base-manifest.json").is_file():
            return {
                "eligible": False,
                "reason": "Pinned model is not acquired; explicit acquisition required",
                "profile": profile,
                "data_counts": artifact["body"]["counts"],
            }
        import psutil
        from llm_lab.masking import preflight
        from llm_lab.model import environment, tokenizer_at, verify_model

        frozen = verify_model(base)
        if frozen["revision"] != profile["revision"] or frozen["provenance"] != profile["model"]:
            raise ValueError("Registered base model identity differs from the selected profile")
        recipe = recipe_for(profile_id)
        masking, _ = preflight(
            files, tokenizer_at(base), recipe.max_length, enable_thinking=recipe.enable_thinking
        )
        free = psutil.disk_usage(str(self.store.root)).free
        ram = psutil.virtual_memory().available
        eligible = free > 2 * 1024**3 and ram > profile["rss_limit_bytes"]
        return {
            "eligible": eligible,
            "reason": "Ready for explicit authorization"
            if eligible
            else "Insufficient measured free memory or disk",
            "profile": profile,
            "recipe": recipe.model_dump(),
            "masking": masking,
            "base_sha256": frozen["base_sha256"],
            "corpus_sha256": manifest["corpus_sha256"],
            "environment": environment(),
            "free_disk_bytes": free,
            "available_ram_bytes": ram,
            "storage": self.store.budget(reserve=2 * 1024**3),
            "output_root": str(self.store.directory(project_id)),
        }

    def profiles(self):
        return list(PROFILES.values())

    def summary(self, project_id):
        from raw_training_labs.jobs import jobs

        project = self.store.project(project_id)
        artifacts = self.artifacts(project_id)
        history = jobs(self.store, project_id)
        completed = {j["kind"] for j in history if j["status"] == "completed"}
        next_action = "Import and review examples"
        if any(a["kind"] == "corpus" for a in artifacts):
            next_action = "Evaluate configured baseline"
        if "baseline" in completed:
            next_action = "Review baseline; train only if behavior needs correction"
        if "train" in completed:
            next_action = "Compare candidate with baseline"
        if "compare" in completed:
            next_action = "Review quality/regressions; export a verified package if useful"
        if "export" in completed:
            next_action = (
                "Delivery package verified; customer acceptance and activation remain separate"
            )
        return {
            "project": project,
            "artifacts": artifacts,
            "jobs": history,
            "next_action": next_action,
        }

    def assessments(self, project_id, job_id):
        self.store.job(project_id, job_id)
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT body FROM assessments WHERE project=? AND job=? ORDER BY created",
                (project_id, job_id),
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def assess(self, project_id, job_id, payload):
        from raw_training_labs.contracts import DecisionReview
        from raw_training_labs.jobs import detail

        review = DecisionReview.model_validate(payload)
        job = detail(self.store, project_id, job_id)
        pairs = job.get("result", {}).get("comparison", {}).get("pairs", [])
        if job["status"] != "completed" or not any(p["id"] == review.case_id for p in pairs):
            raise ValueError("Review requires a retained completed comparison case")
        with self.store.transaction() as db:
            if db.execute("SELECT count(*) FROM assessments").fetchone()[0] >= 10000:
                raise ValueError("Assessment limit reached; deliberate archival is required")
            db.execute(
                "INSERT INTO assessments VALUES(?,?,?,?,?)",
                (new_id(), project_id, job_id, canonical(review.model_dump()), time.time()),
            )
        return review.model_dump()
