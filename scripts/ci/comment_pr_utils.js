"use strict";

function isBotComment(comment) {
  const item = comment && typeof comment === "object" ? comment : {};
  const user = item.user && typeof item.user === "object" ? item.user : {};
  return String(user.type || "") === "Bot";
}

function findBotCommentByMarker(comments, marker) {
  const token = String(marker || "");
  const items = Array.isArray(comments) ? comments : [];
  return (
    items.find(
      (comment) =>
        isBotComment(comment) && String(comment && comment.body ? comment.body : "").includes(token),
    ) || null
  );
}

async function upsertBotIssueComment({
  github,
  owner,
  repo,
  issueNumber,
  body,
  marker,
  perPage = 100,
}) {
  const { data: comments } = await github.rest.issues.listComments({
    owner,
    repo,
    issue_number: issueNumber,
    per_page: perPage,
  });

  const botComment = findBotCommentByMarker(comments, marker);
  if (botComment) {
    const response = await github.rest.issues.updateComment({
      owner,
      repo,
      comment_id: botComment.id,
      body,
    });
    return {
      action: "updated",
      commentId: botComment.id,
      response,
    };
  }

  const response = await github.rest.issues.createComment({
    owner,
    repo,
    issue_number: issueNumber,
    body,
  });
  const created = response && response.data && typeof response.data === "object" ? response.data : {};
  return {
    action: "created",
    commentId: created.id || null,
    response,
  };
}

module.exports = {
  findBotCommentByMarker,
  isBotComment,
  upsertBotIssueComment,
};
