"use strict";

function stringifyCell(value, fallback = "") {
  if (value === undefined || value === null) {
    return fallback;
  }
  return String(value);
}

function markdownTable(headers, rows) {
  const normalizedHeaders = Array.isArray(headers) ? headers.map((item) => stringifyCell(item)) : [];
  const normalizedRows = Array.isArray(rows) ? rows : [];
  const lines = [
    `| ${normalizedHeaders.join(" | ")} |`,
    `|${normalizedHeaders.map(() => "--------").join("|")}|`,
  ];
  for (const row of normalizedRows) {
    const cells = Array.isArray(row) ? row.map((item) => stringifyCell(item)) : [];
    lines.push(`| ${cells.join(" | ")} |`);
  }
  return lines.join("\n");
}

function markdownSection(title, body) {
  return `### ${stringifyCell(title)}\n${stringifyCell(body)}`;
}

function markdownFooter({ updatedAt, sha }) {
  return [
    "---",
    `*Updated: ${stringifyCell(updatedAt)} UTC*`,
    `*Commit: ${stringifyCell(sha)}*`,
  ].join("\n");
}

module.exports = {
  markdownFooter,
  markdownSection,
  markdownTable,
};
