#!/usr/bin/env python3
"""
Evaluation Results Notification System.

Supports multiple notification channels:
- Slack webhooks
- Email (SMTP)
- GitHub Issues (via API)

Usage:
    python3 scripts/notify_eval_results.py --channel slack --threshold-breach
"""

import argparse
import json
import os
import smtplib
import urllib.request
import urllib.error
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class NotificationConfig:
    """Configuration for notification channels."""

    # Environment variable names
    SLACK_WEBHOOK_ENV = "EVAL_SLACK_WEBHOOK"
    EMAIL_SMTP_HOST_ENV = "EVAL_SMTP_HOST"
    EMAIL_SMTP_PORT_ENV = "EVAL_SMTP_PORT"
    EMAIL_FROM_ENV = "EVAL_EMAIL_FROM"
    EMAIL_TO_ENV = "EVAL_EMAIL_TO"
    EMAIL_PASSWORD_ENV = "EVAL_EMAIL_PASSWORD"
    GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
    GITHUB_REPO_ENV = "GITHUB_REPOSITORY"

    @classmethod
    def get_slack_webhook(cls) -> Optional[str]:
        """Get Slack webhook URL from environment."""
        return os.environ.get(cls.SLACK_WEBHOOK_ENV)

    @classmethod
    def get_email_config(cls) -> Optional[Dict]:
        """Get email configuration from environment."""
        host = os.environ.get(cls.EMAIL_SMTP_HOST_ENV)
        if not host:
            return None

        return {
            "host": host,
            "port": int(os.environ.get(cls.EMAIL_SMTP_PORT_ENV, "587")),
            "from": os.environ.get(cls.EMAIL_FROM_ENV, "eval@cadml.local"),
            "to": os.environ.get(cls.EMAIL_TO_ENV, "").split(","),
            "password": os.environ.get(cls.EMAIL_PASSWORD_ENV)
        }

    @classmethod
    def get_github_config(cls) -> Optional[Dict]:
        """Get GitHub configuration from environment."""
        token = os.environ.get(cls.GITHUB_TOKEN_ENV)
        repo = os.environ.get(cls.GITHUB_REPO_ENV)

        if not token or not repo:
            return None

        return {
            "token": token,
            "repo": repo
        }


class EvaluationAnalyzer:
    """Analyze evaluation results for notifications."""

    def __init__(self, history_dir: Path):
        self.history_dir = history_dir
        self.thresholds = {
            "combined": 0.80,
            "vision": 0.65,
            "ocr": 0.90
        }

    def load_latest_results(self) -> Optional[Dict]:
        """Load the most recent evaluation results."""
        combined_files = sorted(self.history_dir.glob("*_combined.json"))
        if not combined_files:
            return None

        latest_file = combined_files[-1]
        with open(latest_file, "r") as f:
            return json.load(f)

    def check_thresholds(self, results: Dict) -> Tuple[bool, List[str]]:
        """Check if results breach any thresholds."""
        breaches = []

        if "combined" in results:
            combined = results["combined"]

            if combined.get("combined_score", 0) < self.thresholds["combined"]:
                breaches.append(
                    f"Combined score ({combined['combined_score']:.3f}) "
                    f"< threshold ({self.thresholds['combined']})"
                )

            if combined.get("vision_score", 0) < self.thresholds["vision"]:
                breaches.append(
                    f"Vision score ({combined['vision_score']:.3f}) "
                    f"< threshold ({self.thresholds['vision']})"
                )

            if combined.get("ocr_score", 0) < self.thresholds["ocr"]:
                breaches.append(
                    f"OCR score ({combined['ocr_score']:.3f}) "
                    f"< threshold ({self.thresholds['ocr']})"
                )

        return len(breaches) > 0, breaches

    def analyze_trend(self, num_records: int = 10) -> Dict:
        """Analyze trend over recent evaluations."""
        combined_files = sorted(self.history_dir.glob("*_combined.json"))[-num_records:]

        if len(combined_files) < 2:
            return {"trend": "insufficient_data"}

        scores = []
        for file in combined_files:
            with open(file, "r") as f:
                data = json.load(f)
                if "combined" in data:
                    scores.append(data["combined"]["combined_score"])

        if not scores:
            return {"trend": "no_scores"}

        # Calculate trend
        recent_avg = sum(scores[-3:]) / len(scores[-3:]) if len(scores) >= 3 else scores[-1]
        older_avg = sum(scores[:-3]) / len(scores[:-3]) if len(scores) > 3 else scores[0]

        trend_direction = "improving" if recent_avg > older_avg else \
                         "declining" if recent_avg < older_avg else "stable"

        return {
            "trend": trend_direction,
            "recent_avg": recent_avg,
            "older_avg": older_avg,
            "change": recent_avg - older_avg,
            "num_records": len(scores)
        }


class NotificationFormatter:
    """Format messages for different channels."""

    @staticmethod
    def format_slack_message(
        results: Dict,
        breaches: List[str],
        trend: Dict,
        report_url: Optional[str] = None
    ) -> Dict:
        """Format message for Slack."""

        # Determine alert level and emoji
        if breaches:
            emoji = "🚨"
            color = "danger"
            title = "Evaluation Threshold Breach Alert"
        elif trend.get("trend") == "declining":
            emoji = "⚠️"
            color = "warning"
            title = "Evaluation Metrics Declining"
        else:
            emoji = "✅"
            color = "good"
            title = "Evaluation Complete"

        # Build message
        combined = results.get("combined", {})
        timestamp = results.get("timestamp", datetime.now(timezone.utc).isoformat())

        attachments = [{
            "color": color,
            "title": f"{emoji} {title}",
            "fields": [
                {
                    "title": "Combined Score",
                    "value": f"{combined.get('combined_score', 0):.3f}",
                    "short": True
                },
                {
                    "title": "Vision Score",
                    "value": f"{combined.get('vision_score', 0):.3f}",
                    "short": True
                },
                {
                    "title": "OCR Score",
                    "value": f"{combined.get('ocr_score', 0):.3f}",
                    "short": True
                },
                {
                    "title": "Trend",
                    "value": f"{trend.get('trend', 'unknown').title()} "
                            f"({trend.get('change', 0):+.3f})",
                    "short": True
                }
            ],
            "footer": "CAD ML Platform",
            "ts": int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp())
        }]

        # Add breaches if any
        if breaches:
            attachments[0]["fields"].append({
                "title": "⚠️ Threshold Breaches",
                "value": "\n".join(f"• {breach}" for breach in breaches),
                "short": False
            })

        # Add report link if available
        if report_url:
            attachments[0]["actions"] = [{
                "type": "button",
                "text": "View Full Report",
                "url": report_url
            }]

        return {
            "text": f"{emoji} CAD ML Evaluation Results",
            "attachments": attachments
        }

    @staticmethod
    def format_email_message(
        results: Dict,
        breaches: List[str],
        trend: Dict,
        report_url: Optional[str] = None
    ) -> Tuple[str, str, str]:
        """Format message for email. Returns (subject, text_body, html_body)."""

        combined = results.get("combined", {})
        timestamp = results.get("timestamp", datetime.now(timezone.utc).isoformat())

        # Determine subject
        if breaches:
            subject = "🚨 [ALERT] CAD ML Evaluation - Threshold Breach"
        elif trend.get("trend") == "declining":
            subject = "⚠️ [WARNING] CAD ML Evaluation - Metrics Declining"
        else:
            subject = "✅ CAD ML Evaluation Report"

        # Text body
        text_body = f"""
CAD ML Platform - Evaluation Report
====================================

Timestamp: {timestamp}
Branch: {results.get('branch', 'unknown')}
Commit: {results.get('commit', 'unknown')}

SCORES
------
Combined Score: {combined.get('combined_score', 0):.3f}
Vision Score:   {combined.get('vision_score', 0):.3f}
OCR Score:      {combined.get('ocr_score', 0):.3f}

TREND
-----
Direction: {trend.get('trend', 'unknown').title()}
Change: {trend.get('change', 0):+.3f}
Recent Average: {trend.get('recent_avg', 0):.3f}
"""

        if breaches:
            text_body += f"""
THRESHOLD BREACHES
------------------
"""
            for breach in breaches:
                text_body += f"• {breach}\n"

        if report_url:
            text_body += f"\nView full report: {report_url}\n"

        # HTML body
        breach_color = "#dc3545" if breaches else "#28a745"
        trend_icon = "📈" if trend.get("trend") == "improving" else \
                    "📉" if trend.get("trend") == "declining" else "➡️"

        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #3b82f6, #6366f1); color: white; padding: 20px; border-radius: 8px 8px 0 0; }}
        .content {{ background: #f9fafb; padding: 20px; border: 1px solid #e5e7eb; border-radius: 0 0 8px 8px; }}
        .scores {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .score-item {{ text-align: center; }}
        .score-value {{ font-size: 2em; font-weight: bold; color: {breach_color}; }}
        .score-label {{ color: #6b7280; margin-top: 5px; }}
        .breach {{ background: #fee2e2; border: 1px solid #fecaca; border-radius: 4px; padding: 10px; margin: 10px 0; }}
        .breach ul {{ margin: 5px 0; padding-left: 20px; }}
        .button {{ display: inline-block; background: #3b82f6; color: white; padding: 10px 20px; text-decoration: none; border-radius: 6px; margin-top: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 CAD ML Evaluation Report</h1>
            <p>{timestamp}</p>
        </div>
        <div class="content">
            <div class="scores">
                <div class="score-item">
                    <div class="score-value">{combined.get('combined_score', 0):.3f}</div>
                    <div class="score-label">Combined</div>
                </div>
                <div class="score-item">
                    <div class="score-value">{combined.get('vision_score', 0):.3f}</div>
                    <div class="score-label">Vision</div>
                </div>
                <div class="score-item">
                    <div class="score-value">{combined.get('ocr_score', 0):.3f}</div>
                    <div class="score-label">OCR</div>
                </div>
            </div>

            <p><strong>Trend:</strong> {trend_icon} {trend.get('trend', 'unknown').title()} ({trend.get('change', 0):+.3f})</p>
"""

        if breaches:
            html_body += """
            <div class="breach">
                <strong>⚠️ Threshold Breaches:</strong>
                <ul>
"""
            for breach in breaches:
                html_body += f"                    <li>{breach}</li>\n"
            html_body += """
                </ul>
            </div>
"""

        if report_url:
            html_body += f"""
            <a href="{report_url}" class="button">View Full Report</a>
"""

        html_body += """
        </div>
    </div>
</body>
</html>
"""

        return subject, text_body, html_body


class NotificationSender:
    """Send notifications to various channels."""

    @staticmethod
    def send_slack(webhook_url: str, message: Dict) -> bool:
        """Send notification to Slack."""
        try:
            data = json.dumps(message).encode("utf-8")
            req = urllib.request.Request(
                webhook_url,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as response:
                return response.status == 200
        except Exception as e:
            print(f"Failed to send Slack notification: {e}")
            return False

    @staticmethod
    def send_email(config: Dict, subject: str, text_body: str, html_body: str) -> bool:
        """Send notification via email."""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = config["from"]
            msg["To"] = ", ".join(config["to"])

            # Add text and HTML parts
            msg.attach(MIMEText(text_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            # Send email
            with smtplib.SMTP(config["host"], config["port"]) as server:
                server.starttls()
                if config.get("password"):
                    server.login(config["from"], config["password"])
                server.send_message(msg)

            return True
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False

    @staticmethod
    def create_github_issue(config: Dict, title: str, body: str) -> bool:
        """Create GitHub issue for threshold breaches."""
        try:
            api_url = f"https://api.github.com/repos/{config['repo']}/issues"
            data = json.dumps({
                "title": title,
                "body": body,
                "labels": ["evaluation", "automated"]
            }).encode("utf-8")

            req = urllib.request.Request(
                api_url,
                data=data,
                headers={
                    "Authorization": f"token {config['token']}",
                    "Content-Type": "application/json",
                    "Accept": "application/vnd.github.v3+json"
                }
            )

            with urllib.request.urlopen(req) as response:
                return response.status == 201
        except Exception as e:
            print(f"Failed to create GitHub issue: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Send evaluation result notifications")
    parser.add_argument("--channel", choices=["slack", "email", "github", "all"],
                        default="all", help="Notification channel(s) to use")
    parser.add_argument("--dir", default="reports/eval_history",
                        help="Directory containing evaluation history")
    parser.add_argument("--threshold-breach-only", action="store_true",
                        help="Only send notifications on threshold breaches")
    parser.add_argument("--report-url", help="URL to the full evaluation report")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print notifications without sending")

    args = parser.parse_args()

    history_dir = Path(args.dir)
    if not history_dir.exists():
        print(f"History directory not found: {history_dir}")
        return 1

    # Analyze evaluation results
    analyzer = EvaluationAnalyzer(history_dir)
    results = analyzer.load_latest_results()

    if not results:
        print("No evaluation results found")
        return 1

    has_breach, breaches = analyzer.check_thresholds(results)
    trend = analyzer.analyze_trend()

    # Check if we should send notification
    if args.threshold_breach_only and not has_breach:
        print("No threshold breaches detected - skipping notification")
        return 0

    # Format messages
    formatter = NotificationFormatter()

    # Send to requested channels
    channels = ["slack", "email", "github"] if args.channel == "all" else [args.channel]
    success_count = 0

    for channel in channels:
        if channel == "slack":
            webhook = NotificationConfig.get_slack_webhook()
            if not webhook:
                print(f"Slack webhook not configured (set {NotificationConfig.SLACK_WEBHOOK_ENV})")
                continue

            message = formatter.format_slack_message(results, breaches, trend, args.report_url)

            if args.dry_run:
                print(f"\n[DRY RUN] Slack message:\n{json.dumps(message, indent=2)}")
            else:
                if NotificationSender.send_slack(webhook, message):
                    print("✅ Slack notification sent")
                    success_count += 1
                else:
                    print("❌ Failed to send Slack notification")

        elif channel == "email":
            config = NotificationConfig.get_email_config()
            if not config:
                print(f"Email not configured (set {NotificationConfig.EMAIL_SMTP_HOST_ENV})")
                continue

            subject, text_body, html_body = formatter.format_email_message(
                results, breaches, trend, args.report_url
            )

            if args.dry_run:
                print(f"\n[DRY RUN] Email:\nSubject: {subject}\n{text_body}")
            else:
                if NotificationSender.send_email(config, subject, text_body, html_body):
                    print(f"✅ Email sent to {', '.join(config['to'])}")
                    success_count += 1
                else:
                    print("❌ Failed to send email")

        elif channel == "github" and has_breach:
            config = NotificationConfig.get_github_config()
            if not config:
                print(f"GitHub not configured (set {NotificationConfig.GITHUB_TOKEN_ENV})")
                continue

            title = f"Evaluation Threshold Breach - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
            body = f"""
## Evaluation Threshold Breach Alert

The latest evaluation has breached configured thresholds:

{chr(10).join(f'- {breach}' for breach in breaches)}

### Current Scores
- Combined: {results.get('combined', {}).get('combined_score', 0):.3f}
- Vision: {results.get('combined', {}).get('vision_score', 0):.3f}
- OCR: {results.get('combined', {}).get('ocr_score', 0):.3f}

### Metadata
- Branch: `{results.get('branch', 'unknown')}`
- Commit: `{results.get('commit', 'unknown')}`
- Timestamp: {results.get('timestamp', 'unknown')}

{"[View Full Report](" + args.report_url + ")" if args.report_url else ""}

*This issue was automatically generated by the evaluation notification system.*
"""

            if args.dry_run:
                print(f"\n[DRY RUN] GitHub Issue:\nTitle: {title}\n{body}")
            else:
                if NotificationSender.create_github_issue(config, title, body):
                    print("✅ GitHub issue created")
                    success_count += 1
                else:
                    print("❌ Failed to create GitHub issue")

    print(f"\nNotifications sent: {success_count}/{len(channels)}")
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    exit(main())