"use strict";

function isOpenIssue(issue) {
  const item = issue && typeof issue === "object" ? issue : {};
  return String(item.state || "").toLowerCase() === "open";
}

function findOpenIssueByMarker(issues, marker) {
  const token = String(marker || "");
  const items = Array.isArray(issues) ? issues : [];
  return (
    items.find(
      (issue) =>
        isOpenIssue(issue) && String(issue && issue.body ? issue.body : "").includes(token),
    ) || null
  );
}

async function upsertOpenIssue({
  github,
  owner,
  repo,
  title,
  body,
  labels = [],
  marker,
  state = "open",
  perPage = 100,
  listLabels = "",
}) {
  const listPayload = {
    owner,
    repo,
    state,
    per_page: perPage,
  };
  if (String(listLabels || "").trim()) {
    listPayload.labels = String(listLabels);
  }

  const { data: issues } = await github.rest.issues.listForRepo(listPayload);
  const existingIssue = findOpenIssueByMarker(issues, marker);
  if (existingIssue) {
    const updatePayload = {
      owner,
      repo,
      issue_number: existingIssue.number,
      title,
      body,
    };
    if (Array.isArray(labels) && labels.length > 0) {
      updatePayload.labels = labels;
    }
    const response = await github.rest.issues.update(updatePayload);
    return {
      action: "updated",
      issueNumber: existingIssue.number,
      response,
    };
  }

  const createPayload = {
    owner,
    repo,
    title,
    body,
  };
  if (Array.isArray(labels) && labels.length > 0) {
    createPayload.labels = labels;
  }
  const response = await github.rest.issues.create(createPayload);
  const created = response && response.data && typeof response.data === "object" ? response.data : {};
  return {
    action: "created",
    issueNumber: created.number || null,
    response,
  };
}

module.exports = {
  findOpenIssueByMarker,
  isOpenIssue,
  upsertOpenIssue,
};
