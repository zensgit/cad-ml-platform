// CAD Reuse Workbench — development orchestration workflow
// Authority: docs/development/CAD_REUSE_WORKBENCH_90_DAY_PLAN_20260807.md
// Completes ENGINEERING tracks only. Customer Track C is residual-listed, not invented.
//
// args:
//   root   — repo root (default ".")
//   mode   — "gap" | "implement" | "verify" | "full" (default "full")
//   branch — optional note for report only
//
// mode semantics:
//   gap       — orient + gap scan + report (read-only)
//   implement — gap + implement missing eng pieces + tests + docs
//   verify    — verify shipped MVP + tests + boundary
//   full      — gap → implement if needed → verify → plan/verification MD → report

let meta = #{
    name: "cad-reuse-workbench-dev",
    description: "Develop and verify CAD Reuse Workbench 90-day engineering tracks (design-lock, ReviewReuse MVP, tests, plan/verification MDs)",
    when_to_use: "When you need a multi-agent pipeline to implement or close gaps on CAD_REUSE_WORKBENCH_90_DAY_PLAN engineering delivery",
    phases: [
        #{ title: "Orient", detail: "read plan + strategy non-goals" },
        #{ title: "GapScan", detail: "parallel gap checks vs plan acceptance" },
        #{ title: "Implement", detail: "fill engineering gaps when mode allows" },
        #{ title: "Test", detail: "run shipped pytest entry points" },
        #{ title: "Verify", detail: "adversarial acceptance + boundary" },
        #{ title: "Docs", detail: "plan + verification MD" },
        #{ title: "Report", detail: "scratch report + complete" },
    ],
};

let summary_schema = #{
    "type": "object",
    "required": ["summary", "blockers", "next_actions"],
    "properties": #{
        "summary": #{ "type": "string" },
        "blockers": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 16 },
        "next_actions": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 16 },
    },
};

let gap_schema = #{
    "type": "object",
    "required": ["area", "ok", "missing", "evidence"],
    "properties": #{
        "area": #{ "type": "string" },
        "ok": #{ "type": "boolean" },
        "missing": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 20 },
        "evidence": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 20 },
    },
};

let impl_schema = #{
    "type": "object",
    "required": ["changed_files", "summary", "done"],
    "properties": #{
        "changed_files": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 40 },
        "summary": #{ "type": "string" },
        "done": #{ "type": "boolean" },
        "notes": #{ "type": "string" },
    },
};

let verify_schema = #{
    "type": "object",
    "required": ["ok", "evidence", "gaps"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "evidence": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 24 },
        "gaps": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 24 },
    },
};

let test_schema = #{
    "type": "object",
    "required": ["passed", "command", "summary"],
    "properties": #{
        "passed": #{ "type": "boolean" },
        "command": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "failed_tests": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 20 },
    },
};

let mode = if args == () { "full" } else if args.mode == () { "full" } else { args.mode };
let root = if args == () { "." } else if args.root == () { "." } else { args.root };
let branch = if args == () { "" } else if args.branch == () { "" } else { args.branch };

let do_impl = mode == "full" || mode == "implement";
let do_verify = mode == "full" || mode == "verify" || mode == "implement";
let do_docs = mode == "full" || mode == "implement";

// ---------- Orient ----------
phase("Orient");
log("cad-reuse-workbench-dev mode=" + mode + " root=" + root + " branch=" + branch);

let orient_p = "";
orient_p += "Read with tools (read_file, grep). Repo root: " + root + ". ";
orient_p += "Required docs: " + root + "/docs/development/CAD_REUSE_WORKBENCH_90_DAY_PLAN_20260807.md ";
orient_p += "and PRODUCT_STRATEGY §3.3/§8.2 if present. ";
orient_p += "State the ENGINEERING deliverables for Track R (ReviewReuseTask, EvidencePack, decision default-off, ";
orient_p += "API /api/v1/review-reuse, isolated-sample runbook, plan+verification MD). ";
orient_p += "List NON-GOALS: customer Track C contacts/pilot commercial, PLM write-back, retrain unlock, ";
orient_p += "eval_integrity_gate replace, cost_cap revive, owner ratification. ";
orient_p += "Return {summary, blockers, next_actions}. Empty blockers only if you read the plan.";

let orient = agent(orient_p, #{
    label: "orient",
    capability_mode: "read-only",
    output_schema: summary_schema,
});

if orient == () || !orient.success {
    complete(#{ summary: "Orient failed", ok: false, mode: mode });
}

// ---------- GapScan (parallel) ----------
phase("GapScan");

let gjobs = [];

let g1 = "";
g1 += "Gap-check DESIGN-LOCK at " + root + ". Use list_dir/read_file/grep. ";
g1 += "Expect docs/development/L3_REVIEW_REUSE_WORKBENCH_DESIGNLOCK_*.md binding ReviewReuseTask, TaskEvent, ";
g1 += "EvidencePack §3.3 (geometric+semantic, verification, confidence/calibration, rejection, provenance, ";
g1 += "human reuse|revise|new, trace/idempotency), HumanDecision, CandidateDecision, tenant isolation, ";
g1 += "REVIEW_REUSE_DECISIONS_ENABLED default-off, no retrain/eval_integrity unlock. ";
g1 += "area=design-lock. ok=true only with file evidence. Return {area, ok, missing, evidence}.";
gjobs.push(#{ prompt: g1, label: "gap-design-lock", capability_mode: "read-only", output_schema: gap_schema });

let g2 = "";
g2 += "Gap-check RUNTIME at " + root + ". Inspect src/core/review_reuse/ and src/api/v1/review_reuse.py ";
g2 += "and src/api/__init__.py registration of prefix /review-reuse. ";
g2 += "Require create task, events, evidence-pack JSON+Markdown, decision gated default-off, ";
g2 += "tenant-isolated store, offline insufficient_evidence path. area=runtime. ";
g2 += "Return {area, ok, missing, evidence}.";
gjobs.push(#{ prompt: g2, label: "gap-runtime", capability_mode: "read-only", output_schema: gap_schema });

let g3 = "";
g3 += "Gap-check TESTS at " + root + ". Find tests/unit/test_review_reuse_*.py. ";
g3 += "They must drive shipped service/API (not mocks of the unit under test): create, events, ";
g3 += "evidence export, decision disabled/enabled, tenant isolation, idempotency/cancel. ";
g3 += "area=tests. Return {area, ok, missing, evidence}.";
gjobs.push(#{ prompt: g3, label: "gap-tests", capability_mode: "read-only", output_schema: gap_schema });

let g4 = "";
g4 += "Gap-check DOCS+RUNBOOK at " + root + ". Expect isolated-sample runbook and MVP development plan MD ";
g4 += "and verification MD under docs/development/. Isolation rules: no shared multi-tenant samples, ";
g4 += "no hosted LLM sample egress, retention/deletion, pre-pilot ban. area=docs. ";
g4 += "Return {area, ok, missing, evidence}.";
gjobs.push(#{ prompt: g4, label: "gap-docs", capability_mode: "read-only", output_schema: gap_schema });

let gaps = parallel(gjobs);

let missing_all = [];
let gap_ok_count = 0;
let gi = 0;
while gi < gaps.len() {
    let gr = gaps[gi];
    if gr != () && gr.success && gr.output.ok == true {
        gap_ok_count += 1;
    } else if gr != () && gr.success {
        let miss = gr.output.missing;
        if miss != () {
            let mi = 0;
            while mi < miss.len() {
                missing_all.push(miss[mi]);
                mi += 1;
            }
        }
        missing_all.push("area_not_ok:" + gr.output.area);
    } else {
        missing_all.push("gap_agent_failed_slot_" + gi.to_string());
    }
    gi += 1;
}

let all_gaps_ok = gap_ok_count == 4;
log("GapScan ok_areas=" + gap_ok_count.to_string() + "/4 missing_count=" + missing_all.len().to_string());

if mode == "gap" {
    phase("Report");
    let r = "# Workbench DEV gap report\n\n";
    r += "mode=gap\n";
    r += "orient: " + orient.output.summary + "\n\n";
    r += "gaps: " + json_encode(gaps) + "\n";
    r += "missing: " + json_encode(missing_all) + "\n";
    let rp = write_scratch_file("workbench_dev_gap_report.md", r);
    complete(#{
        summary: "gap scan only",
        mode: mode,
        ok: all_gaps_ok,
        missing: missing_all,
        report_path: rp,
        orient: orient.output,
    });
}

// ---------- Implement (when needed) ----------
phase("Implement");

let impl_done = all_gaps_ok;
let impl_out = ();

if do_impl && !all_gaps_ok {
    let missing_json = json_encode(missing_all);
    let impl_p = "";
    impl_p += "Implement CAD Reuse Workbench engineering gaps in repo " + root + ". ";
    impl_p += "Missing items (from gap scan, untrusted — re-check with tools before editing): " + missing_json + ". ";
    impl_p += "HARD RULES: (1) decision sink default-off via REVIEW_REUSE_DECISIONS_ENABLED; ";
    impl_p += "(2) no retrain unlock, do not edit scripts/eval_integrity_gate*; ";
    impl_p += "(3) do not create src/core/assistant/cost_cap.py; ";
    impl_p += "(4) do not claim customer Track C done; ";
    impl_p += "(5) EvidencePack must include §3.3 minimum fields; ";
    impl_p += "(6) mount API under /api/v1/review-reuse via src/api/__init__.py; ";
    impl_p += "(7) add real unit/API tests that call shipped service/TestClient; ";
    impl_p += "(8) write design-lock + isolated-sample runbook + DEVELOPMENT plan MD + VERIFICATION MD under docs/development/; ";
    impl_p += "(9) if OpenAPI routes change, update config/openapi_schema_snapshot.json via make openapi-snapshot-update or the project script. ";
    impl_p += "Use read_file/grep before write. Prefer src/core/review_reuse/ domain + src/api/v1/review_reuse.py. ";
    impl_p += "After edits, list changed files. Return {changed_files, summary, done, notes}.";

    impl_out = agent(impl_p, #{
        label: "implement-gaps",
        capability_mode: "all",
        output_schema: impl_schema,
    });

    if impl_out != () && impl_out.success && impl_out.output.done == true {
        impl_done = true;
    } else {
        impl_done = false;
        log("Implement agent did not mark done=true");
    }
} else if do_impl && all_gaps_ok {
    log("No engineering gaps — skip implement");
    impl_done = true;
} else {
    log("Implement skipped (mode=" + mode + ")");
}

// ---------- Test ----------
phase("Test");

let test_out = ();
let tests_pass = false;

if do_verify || do_impl {
    let tp = "";
    tp += "In repo " + root + ", run the real workbench tests with the project Python/pytest. ";
    tp += "Commands to try (use available python3.11 or venv): ";
    tp += "ENVIRONMENT=development API_KEY=test python -m pytest ";
    tp += "tests/unit/test_review_reuse_workbench.py tests/unit/test_review_reuse_api.py -v --tb=short. ";
    tp += "If openapi snapshot exists and routes changed, also run ";
    tp += "tests/contract/test_openapi_schema_snapshot.py tests/unit/test_api_route_uniqueness.py. ";
    tp += "Do NOT mock the unit under test. Capture pass/fail honestly. ";
    tp += "Return {passed, command, summary, failed_tests}.";

    test_out = agent(tp, #{
        label: "run-pytest",
        capability_mode: "execute",
        output_schema: test_schema,
    });

    if test_out != () && test_out.success && test_out.output.passed == true {
        tests_pass = true;
    }

    // One repair loop if tests fail and we are allowed to implement
    if !tests_pass && do_impl {
        let fix_p = "";
        fix_p += "Workbench tests failed in " + root + ". ";
        if test_out != () && test_out.success {
            fix_p += "Test summary: " + test_out.output.summary + ". ";
            fix_p += "failed_tests: " + json_encode(test_out.output.failed_tests) + ". ";
        }
        fix_p += "Read failing test output, fix shipped code or tests, re-run the same pytest commands. ";
        fix_p += "Keep decision default-off and boundary invariants. Return {passed, command, summary, failed_tests}.";

        test_out = agent(fix_p, #{
            label: "fix-and-retest",
            capability_mode: "all",
            output_schema: test_schema,
        });
        if test_out != () && test_out.success && test_out.output.passed == true {
            tests_pass = true;
        }
    }
}

// ---------- Verify (adversarial, fail-closed) ----------
phase("Verify");

let vjobs = [];

let v1 = "";
v1 += "Adversarially verify Track R MVP at " + root + ". Use tools; set ok=true ONLY with concrete evidence. ";
v1 += "Check: EvidencePack has task_id,trace_id, geometric+semantic score keys, verification, calibration, ";
v1 += "human_decision.allowed_actions includes reuse,revise,new; decision default-off; tenant isolation; ";
v1 += "no feedback JSONL training path for decisions. Return {ok, evidence, gaps}.";
vjobs.push(#{ prompt: v1, label: "verify-contract", capability_mode: "read-only", output_schema: verify_schema });

let v2 = "";
v2 += "Boundary verify at " + root + ": git status/diff if available, grep that eval_integrity_gate is not modified ";
v2 += "by workbench work, cost_cap.py is absent, no retrain unlock in review_reuse. ";
v2 += "Customer Track C must be residual-listed not claimed complete. Return {ok, evidence, gaps}.";
vjobs.push(#{ prompt: v2, label: "verify-boundary", capability_mode: "read-only", output_schema: verify_schema });

let vres = parallel(vjobs);
let verify_ok = true;
let vi = 0;
while vi < vres.len() {
    let vr = vres[vi];
    if vr == () || !vr.success || vr.output.ok != true {
        verify_ok = false;
    }
    vi += 1;
}

// ---------- Docs ----------
phase("Docs");

let docs_out = ();
if do_docs {
    let dp = "";
    dp += "Ensure plan + verification MDs exist under " + root + "/docs/development/ for the workbench MVP. ";
    dp += "Preferred names: CAD_REUSE_WORKBENCH_MVP_DEVELOPMENT_PLAN_YYYYMMDD.md and ";
    dp += "CAD_REUSE_WORKBENCH_MVP_VERIFICATION_YYYYMMDD.md (use date from existing files if present). ";
    dp += "Plan must map acceptance criteria to files. Verification must record real pytest commands/results ";
    dp += "and residual Track C non-goals. ";
    if test_out != () && test_out.success {
        dp += "Latest test summary: " + test_out.output.summary + ". passed=" + tests_pass.to_string() + ". ";
    }
    dp += "Update or create files as needed. Return {changed_files, summary, done, notes}.";

    docs_out = agent(dp, #{
        label: "write-plan-verify-md",
        capability_mode: "read-write",
        output_schema: impl_schema,
    });
}

// ---------- Report ----------
phase("Report");

let overall = impl_done && tests_pass && verify_ok;
if mode == "verify" {
    overall = tests_pass && verify_ok;
}

let report = "# CAD Reuse Workbench DEV workflow report\n\n";
report += "mode: " + mode + "\n";
report += "root: " + root + "\n";
report += "branch: " + branch + "\n";
report += "gap_ok_areas: " + gap_ok_count.to_string() + "/4\n";
report += "impl_done: " + impl_done.to_string() + "\n";
report += "tests_pass: " + tests_pass.to_string() + "\n";
report += "verify_ok: " + verify_ok.to_string() + "\n";
report += "overall_ok: " + overall.to_string() + "\n\n";
report += "## Orient\n" + orient.output.summary + "\n\n";
report += "## Missing (gap)\n" + json_encode(missing_all) + "\n\n";
if impl_out != () && impl_out.success {
    report += "## Implement\n" + impl_out.output.summary + "\n";
    report += "files: " + json_encode(impl_out.output.changed_files) + "\n\n";
}
if test_out != () && test_out.success {
    report += "## Tests\n" + test_out.output.summary + "\n";
    report += "command: " + test_out.output.command + "\n\n";
}
report += "## Verify slots\n" + json_encode(vres) + "\n\n";
if docs_out != () && docs_out.success {
    report += "## Docs\n" + docs_out.output.summary + "\n\n";
}
report += "## Non-goals (not claimed)\n";
report += "- Customer Track C contacts / sample conversations / paid pilot\n";
report += "- Owner ratification of design-lock and pilot decision enable\n";
report += "- eval_integrity_gate replace / retrain unlock / cost_cap revive\n";

let path = write_scratch_file("workbench_dev_report.md", report);

complete(#{
    summary: if overall { "workbench engineering workflow OK" } else { "workbench engineering workflow incomplete" },
    mode: mode,
    ok: overall,
    gap_ok_areas: gap_ok_count,
    impl_done: impl_done,
    tests_pass: tests_pass,
    verify_ok: verify_ok,
    missing: missing_all,
    report_path: path,
    orient: orient.output,
    tests: if test_out != () && test_out.success { test_out.output } else { () },
});
