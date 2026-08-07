let meta = #{
    name: "cad-reuse-workbench-90d",
    description: "Orchestrate CAD Reuse Workbench 90-day plan tracks: design-lock, MVP verify, safety audit residuals, customer residual checklist",
    when_to_use: "When executing or re-verifying the CAD_REUSE_WORKBENCH_90_DAY_PLAN engineering + residual tracks",
    phases: [
        #{ title: "Orient", detail: "read plan + strategy boundaries" },
        #{ title: "VerifyR", detail: "Track R workbench MVP evidence" },
        #{ title: "Safety", detail: "Track E/S residual audit prompts" },
        #{ title: "Residual", detail: "Track C/O residual checklist" },
        #{ title: "Report", detail: "write plan/verification summary" },
    ],
};

let summary_schema = #{
    "type": "object",
    "required": ["summary", "blockers", "next_actions"],
    "properties": #{
        "summary": #{ "type": "string" },
        "blockers": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 12 },
        "next_actions": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 12 },
    },
};

let verify_schema = #{
    "type": "object",
    "required": ["ok", "evidence", "gaps"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "evidence": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 20 },
        "gaps": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 20 },
    },
};

// args.track: "r-mvp" (default) | "full" | "safety" | "residual"
let track = if args == () { "r-mvp" } else if args.track == () { "r-mvp" } else { args.track };
let root = if args == () { "." } else if args.root == () { "." } else { args.root };

phase("Orient");
log("cad-reuse-workbench-90d track=" + track + " root=" + root);

let orient_prompt = "";
orient_prompt += "You are verifying the CAD Reuse Workbench 90-day engineering track. ";
orient_prompt += "Read with tools (read_file/grep), do not invent: ";
orient_prompt += root + "/docs/development/CAD_REUSE_WORKBENCH_90_DAY_PLAN_20260807.md, ";
orient_prompt += root + "/docs/development/L3_REVIEW_REUSE_WORKBENCH_DESIGNLOCK_20260808.md if present, ";
orient_prompt += root + "/docs/PRODUCT_STRATEGY.md sections 3.3 and 8.2 if needed. ";
orient_prompt += "Confirm boundaries: no retrain unlock, no eval_integrity_gate replace, decision default-off, Day90=2026-10-10. ";
orient_prompt += "Return {summary, blockers, next_actions}.";

let orient = agent(orient_prompt, #{
    label: "orient",
    capability_mode: "read-only",
    output_schema: summary_schema,
});

if orient == () || !orient.success {
    complete(#{
        summary: "Orient agent failed",
        track: track,
        ok: false,
    });
}

phase("VerifyR");
let r_prompt = "";
r_prompt += "Verify Track R ReviewReuse MVP in the repo at " + root + ". ";
r_prompt += "Use read_file/grep/list_dir. Check that these exist and match the plan: ";
r_prompt += "src/core/review_reuse/, src/api/v1/review_reuse.py, registration of /review-reuse, ";
r_prompt += "REVIEW_REUSE_DECISIONS_ENABLED default-off, tests under tests/unit/test_review_reuse_*.py, ";
r_prompt += "EvidencePack §3.3 fields in evidence.py. ";
r_prompt += "Set ok=true only with concrete file evidence. Return {ok, evidence, gaps}.";

let r_verify = agent(r_prompt, #{
    label: "verify-track-r",
    capability_mode: "read-only",
    output_schema: verify_schema,
});

let r_ok = r_verify != () && r_verify.success && r_verify.output.ok == true;

if track == "r-mvp" {
    phase("Report");
    let report = "# CAD Reuse Workbench workflow report (r-mvp)\n\n";
    report += "## Orient\n" + orient.output.summary + "\n\n";
    report += "## Track R ok=" + r_ok.to_string() + "\n";
    if r_verify != () && r_verify.success {
        report += "evidence: " + json_encode(r_verify.output.evidence) + "\n";
        report += "gaps: " + json_encode(r_verify.output.gaps) + "\n";
    }
    let path = write_scratch_file("workbench_r_mvp_report.md", report);
    complete(#{
        summary: "r-mvp verification complete",
        track: track,
        ok: r_ok,
        report_path: path,
        orient: orient.output,
        track_r: if r_verify != () && r_verify.success { r_verify.output } else { () },
    });
}

phase("Safety");
let s_jobs = [];
s_jobs.push(#{
    prompt: "Audit Track E E1 residual at " + root + ": read scripts/track_e_eval_integrity.py and docs mentioning Track E. Confirm dry-run / no retrain unlock / no model-run metrics claim. Return {ok, evidence, gaps}.",
    label: "audit-e1",
    capability_mode: "read-only",
    output_schema: verify_schema,
});
s_jobs.push(#{
    prompt: "Audit preserved safety baselines at " + root + ": SEAL (#537) and production identity (#538). Grep for ASSISTANT_HOSTED_PROVIDER_OPT_IN and production identity fail-closed. Confirm no cost_cap revival. Return {ok, evidence, gaps}.",
    label: "audit-seal-identity",
    capability_mode: "read-only",
    output_schema: verify_schema,
});
let s_results = parallel(s_jobs);

if track == "safety" {
    phase("Report");
    let sreport = "# Safety residual report\n";
    let path2 = write_scratch_file("workbench_safety_report.md", sreport + json_encode(s_results));
    complete(#{ summary: "safety track done", track: track, results: s_results, report_path: path2, ok: r_ok });
}

phase("Residual");
let c_prompt = "";
c_prompt += "Track C/O residual checklist for CAD reuse workbench at " + root + ". ";
c_prompt += "Read the 90-day plan customer sections and isolated sample runbook if present. ";
c_prompt += "List residual customer actions (contacts, samples, pilot metrics) and ops package items still open. ";
c_prompt += "Return {summary, blockers, next_actions}. Do not claim customer work is done without evidence.";

let residual = agent(c_prompt, #{
    label: "track-c-o-residual",
    capability_mode: "read-only",
    output_schema: summary_schema,
});

phase("Report");
let full = "# CAD Reuse Workbench 90d workflow report\n\n";
full += "track=" + track + "\n\n";
full += "## Orient\n" + orient.output.summary + "\n\n";
full += "## Track R ok=" + r_ok.to_string() + "\n";
if residual != () && residual.success {
    full += "## Residual\n" + residual.output.summary + "\n";
}
let fpath = write_scratch_file("workbench_90d_report.md", full);
complete(#{
    summary: "full/residual workflow complete",
    track: track,
    ok: r_ok,
    report_path: fpath,
    orient: orient.output,
    track_r: if r_verify != () && r_verify.success { r_verify.output } else { () },
    residual: if residual != () && residual.success { residual.output } else { () },
});
