#!/usr/bin/env python3
"""
治理报告调度器
Governance Report Scheduler - 自动化月度/周度/日度报告生成

Features:
1. 多种调度模式（cron、systemd、APScheduler）
2. 报告分发（Email、Slack、S3）
3. 失败重试与告警
4. 报告存档管理
"""

import os
import sys
import json
import time
import logging
import argparse
import smtplib
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    HAS_APSCHEDULER = True
except ImportError:
    HAS_APSCHEDULER = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

from monthly_governance_report import MonthlyGovernanceReport

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/governance_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ReportDistributor:
    """报告分发器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化分发器

        Args:
            config: 分发配置
        """
        self.config = config
        self.email_config = config.get('email', {})
        self.slack_config = config.get('slack', {})
        self.s3_config = config.get('s3', {})

    def distribute(self, report_path: Path, report_type: str = "monthly") -> bool:
        """
        分发报告

        Args:
            report_path: 报告文件路径
            report_type: 报告类型

        Returns:
            bool: 是否成功
        """
        success = True

        # Email分发
        if self.email_config.get('enabled'):
            try:
                self._send_email(report_path, report_type)
                logger.info(f"Email sent successfully for {report_type} report")
            except Exception as e:
                logger.error(f"Failed to send email: {e}")
                success = False

        # Slack分发
        if self.slack_config.get('enabled') and HAS_REQUESTS:
            try:
                self._send_slack(report_path, report_type)
                logger.info(f"Slack notification sent for {report_type} report")
            except Exception as e:
                logger.error(f"Failed to send Slack: {e}")
                success = False

        # S3上传
        if self.s3_config.get('enabled') and HAS_BOTO3:
            try:
                self._upload_s3(report_path, report_type)
                logger.info(f"S3 upload successful for {report_type} report")
            except Exception as e:
                logger.error(f"Failed to upload to S3: {e}")
                success = False

        return success

    def _send_email(self, report_path: Path, report_type: str):
        """发送邮件"""
        smtp_server = self.email_config.get('smtp_server')
        smtp_port = self.email_config.get('smtp_port', 587)
        username = self.email_config.get('username')
        password = self.email_config.get('password')
        recipients = self.email_config.get('recipients', [])

        if not all([smtp_server, username, password, recipients]):
            raise ValueError("Email configuration incomplete")

        # 创建邮件
        msg = MIMEMultipart()
        msg['From'] = username
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = f"CAD ML Platform {report_type.title()} Governance Report - {datetime.now().strftime('%Y-%m')}"

        # 添加正文
        with open(report_path, 'r', encoding='utf-8') as f:
            body = f.read()

        # 提取摘要
        summary = self._extract_summary(body)
        msg.attach(MIMEText(summary, 'plain'))

        # 添加附件
        attachment = MIMEBase('application', 'octet-stream')
        with open(report_path, 'rb') as f:
            attachment.set_payload(f.read())
        encoders.encode_base64(attachment)
        attachment.add_header(
            'Content-Disposition',
            f'attachment; filename="{report_path.name}"'
        )
        msg.attach(attachment)

        # 发送邮件
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)

    def _send_slack(self, report_path: Path, report_type: str):
        """发送Slack通知"""
        webhook_url = self.slack_config.get('webhook_url')
        if not webhook_url:
            raise ValueError("Slack webhook URL not configured")

        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取关键信息
        summary = self._extract_summary(content)

        # 构建Slack消息
        message = {
            "text": f"📊 {report_type.title()} Governance Report",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"CAD ML Platform {report_type.title()} Report"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": summary
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"📄 Full report: {self._get_report_url(report_path)}"
                    }
                }
            ]
        }

        response = requests.post(webhook_url, json=message)
        response.raise_for_status()

    def _upload_s3(self, report_path: Path, report_type: str):
        """上传到S3"""
        bucket = self.s3_config.get('bucket')
        prefix = self.s3_config.get('prefix', 'governance-reports')

        if not bucket:
            raise ValueError("S3 bucket not configured")

        s3_client = boto3.client('s3')

        # 构建S3 key
        timestamp = datetime.now().strftime('%Y%m%d')
        s3_key = f"{prefix}/{report_type}/{timestamp}/{report_path.name}"

        # 上传文件
        s3_client.upload_file(
            str(report_path),
            bucket,
            s3_key,
            ExtraArgs={
                'ContentType': 'text/markdown',
                'Metadata': {
                    'report_type': report_type,
                    'generated_at': datetime.now().isoformat()
                }
            }
        )

    def _extract_summary(self, content: str) -> str:
        """提取报告摘要"""
        lines = content.split('\n')
        summary_lines = []
        in_summary = False

        for line in lines:
            if '执行摘要' in line or 'Executive Summary' in line:
                in_summary = True
                continue
            elif in_summary and line.startswith('#'):
                break
            elif in_summary:
                summary_lines.append(line)

        return '\n'.join(summary_lines[:20])  # 限制摘要长度

    def _get_report_url(self, report_path: Path) -> str:
        """获取报告URL"""
        if self.s3_config.get('enabled'):
            bucket = self.s3_config.get('bucket')
            region = self.s3_config.get('region', 'us-east-1')
            prefix = self.s3_config.get('prefix', 'governance-reports')
            timestamp = datetime.now().strftime('%Y%m%d')
            return f"https://{bucket}.s3.{region}.amazonaws.com/{prefix}/monthly/{timestamp}/{report_path.name}"
        return str(report_path.absolute())


class GovernanceScheduler:
    """治理报告调度器"""

    def __init__(self, config_file: Optional[str] = None):
        """
        初始化调度器

        Args:
            config_file: 配置文件路径
        """
        self.config = self._load_config(config_file)
        self.distributor = ReportDistributor(self.config.get('distribution', {}))
        self.report_generator = MonthlyGovernanceReport()
        self.scheduler = None

        # 初始化调度器
        if HAS_APSCHEDULER and self.config.get('scheduler', {}).get('type') == 'apscheduler':
            self.scheduler = BackgroundScheduler()

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """加载配置"""
        default_config = {
            'scheduler': {
                'type': 'cron',  # cron, apscheduler, or simple
                'monthly': {
                    'enabled': True,
                    'day': 1,
                    'hour': 2,
                    'minute': 0
                },
                'weekly': {
                    'enabled': True,
                    'weekday': 1,  # Monday
                    'hour': 9,
                    'minute': 0
                },
                'daily': {
                    'enabled': False,
                    'hour': 6,
                    'minute': 0
                }
            },
            'distribution': {
                'email': {
                    'enabled': False,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'username': '',
                    'password': '',
                    'recipients': []
                },
                'slack': {
                    'enabled': False,
                    'webhook_url': ''
                },
                's3': {
                    'enabled': False,
                    'bucket': '',
                    'prefix': 'governance-reports',
                    'region': 'us-east-1'
                }
            },
            'archive': {
                'enabled': True,
                'retention_days': 90,
                'compress': True
            },
            'retry': {
                'max_attempts': 3,
                'delay_seconds': 300
            }
        }

        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                # 深度合并配置
                self._deep_merge(default_config, user_config)

        return default_config

    def _deep_merge(self, base: dict, override: dict):
        """深度合并字典"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def generate_monthly_report(self) -> Optional[Path]:
        """生成月度报告"""
        try:
            logger.info("Starting monthly report generation")

            # 生成报告
            report_path = self.report_generator.generate_report()
            logger.info(f"Monthly report generated: {report_path}")

            # 分发报告
            if self.distributor.distribute(report_path, "monthly"):
                logger.info("Monthly report distributed successfully")
            else:
                logger.warning("Some distribution channels failed")

            # 归档管理
            self._archive_old_reports()

            return report_path

        except Exception as e:
            logger.error(f"Failed to generate monthly report: {e}")
            return None

    def generate_weekly_summary(self) -> Optional[Path]:
        """生成周度摘要"""
        try:
            logger.info("Starting weekly summary generation")

            # 简化版报告，只包含关键指标
            report_data = self.report_generator._collect_governance_data()

            # 创建周度摘要
            summary_path = Path(f"reports/weekly_summary_{datetime.now().strftime('%Y%m%d')}.md")
            summary_path.parent.mkdir(exist_ok=True)

            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(self._generate_weekly_summary(report_data))

            logger.info(f"Weekly summary generated: {summary_path}")

            # 分发
            self.distributor.distribute(summary_path, "weekly")

            return summary_path

        except Exception as e:
            logger.error(f"Failed to generate weekly summary: {e}")
            return None

    def _generate_weekly_summary(self, data: Dict[str, Any]) -> str:
        """生成周度摘要内容"""
        overall_score = self.report_generator._calculate_overall_score(data)

        summary = f"""# CAD ML Platform 周度治理摘要
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 核心指标
- **总体合规评分**: {overall_score:.1f}%
- **错误码覆盖率**: {data['error_codes']['coverage']:.1f}%
- **韧性层健康度**: {data['resilience']['health_score']:.1f}%
- **指标基数使用**: {data['cardinality']['usage_percentage']:.1f}%

## 本周关键事件
- 熔断触发: {data['resilience']['circuit_breaker_trips']}次
- 限流事件: {data['resilience']['rate_limit_hits']}次
- 新增错误码: {len(data['error_codes'].get('new_codes', []))}个
- 指标漂移: {len(data['drift'].get('detected_drifts', []))}个

## 需要关注
"""
        # 添加警告项
        if data['cardinality']['usage_percentage'] > 80:
            summary += f"- ⚠️ 指标基数使用率高: {data['cardinality']['usage_percentage']:.1f}%\n"

        if data['resilience']['health_score'] < 85:
            summary += f"- ⚠️ 韧性层健康度低: {data['resilience']['health_score']:.1f}%\n"

        return summary

    def _archive_old_reports(self):
        """归档旧报告"""
        if not self.config['archive']['enabled']:
            return

        retention_days = self.config['archive']['retention_days']
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        reports_dir = Path('reports')
        archive_dir = Path('reports/archive')
        archive_dir.mkdir(exist_ok=True)

        for report_file in reports_dir.glob('*.md'):
            # 检查文件时间
            file_time = datetime.fromtimestamp(report_file.stat().st_mtime)
            if file_time < cutoff_date:
                # 移动到归档目录
                archive_path = archive_dir / report_file.name
                report_file.rename(archive_path)

                # 压缩如果启用
                if self.config['archive']['compress']:
                    subprocess.run(['gzip', str(archive_path)])

                logger.info(f"Archived old report: {report_file.name}")

    def setup_cron(self):
        """设置cron调度"""
        cron_entries = []

        # 月度报告
        if self.config['scheduler']['monthly']['enabled']:
            day = self.config['scheduler']['monthly']['day']
            hour = self.config['scheduler']['monthly']['hour']
            minute = self.config['scheduler']['monthly']['minute']
            script_path = Path(__file__).absolute()
            cron_entries.append(
                f"{minute} {hour} {day} * * cd {script_path.parent} && python3 {script_path} --generate monthly"
            )

        # 周度报告
        if self.config['scheduler']['weekly']['enabled']:
            weekday = self.config['scheduler']['weekly']['weekday']
            hour = self.config['scheduler']['weekly']['hour']
            minute = self.config['scheduler']['weekly']['minute']
            script_path = Path(__file__).absolute()
            cron_entries.append(
                f"{minute} {hour} * * {weekday} cd {script_path.parent} && python3 {script_path} --generate weekly"
            )

        # 输出cron配置
        print("# Add these lines to your crontab (crontab -e):")
        for entry in cron_entries:
            print(entry)

    def setup_systemd(self):
        """生成systemd服务和timer配置"""
        script_path = Path(__file__).absolute()

        # Service文件
        service_content = f"""[Unit]
Description=CAD ML Platform Governance Report Generator
After=network.target

[Service]
Type=oneshot
WorkingDirectory={script_path.parent}
ExecStart=/usr/bin/python3 {script_path} --generate monthly
User={os.getenv('USER', 'nobody')}
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""

        # Timer文件
        timer_content = f"""[Unit]
Description=Monthly Governance Report Timer
Requires=governance-report.service

[Timer]
OnCalendar=monthly
OnCalendar=*-*-01 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
"""

        print("# Save these as systemd units:")
        print("\n# /etc/systemd/system/governance-report.service:")
        print(service_content)
        print("\n# /etc/systemd/system/governance-report.timer:")
        print(timer_content)
        print("\n# Then run:")
        print("# sudo systemctl daemon-reload")
        print("# sudo systemctl enable governance-report.timer")
        print("# sudo systemctl start governance-report.timer")

    def run_scheduler(self):
        """运行调度器"""
        if self.scheduler and HAS_APSCHEDULER:
            # APScheduler模式
            logger.info("Starting APScheduler")

            # 月度报告
            if self.config['scheduler']['monthly']['enabled']:
                self.scheduler.add_job(
                    self.generate_monthly_report,
                    CronTrigger(
                        day=self.config['scheduler']['monthly']['day'],
                        hour=self.config['scheduler']['monthly']['hour'],
                        minute=self.config['scheduler']['monthly']['minute']
                    ),
                    id='monthly_report'
                )

            # 周度报告
            if self.config['scheduler']['weekly']['enabled']:
                self.scheduler.add_job(
                    self.generate_weekly_summary,
                    CronTrigger(
                        day_of_week=self.config['scheduler']['weekly']['weekday'],
                        hour=self.config['scheduler']['weekly']['hour'],
                        minute=self.config['scheduler']['weekly']['minute']
                    ),
                    id='weekly_summary'
                )

            self.scheduler.start()

            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                logger.info("Shutting down scheduler")
                self.scheduler.shutdown()

        elif HAS_SCHEDULE:
            # Simple schedule模式
            logger.info("Starting simple scheduler")

            # 月度报告
            if self.config['scheduler']['monthly']['enabled']:
                schedule.every().month.at(
                    f"{self.config['scheduler']['monthly']['hour']:02d}:{self.config['scheduler']['monthly']['minute']:02d}"
                ).do(self.generate_monthly_report)

            # 周度报告
            if self.config['scheduler']['weekly']['enabled']:
                schedule.every().week.at(
                    f"{self.config['scheduler']['weekly']['hour']:02d}:{self.config['scheduler']['weekly']['minute']:02d}"
                ).do(self.generate_weekly_summary)

            while True:
                schedule.run_pending()
                time.sleep(60)
        else:
            logger.error("No scheduler library available. Install 'schedule' or 'apscheduler'")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CAD ML Platform Governance Report Scheduler')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--generate', choices=['monthly', 'weekly', 'daily'],
                       help='Generate report immediately')
    parser.add_argument('--setup', choices=['cron', 'systemd', 'apscheduler'],
                       help='Setup scheduling method')
    parser.add_argument('--run', action='store_true', help='Run scheduler daemon')

    args = parser.parse_args()

    scheduler = GovernanceScheduler(args.config)

    if args.generate:
        # 立即生成报告
        if args.generate == 'monthly':
            scheduler.generate_monthly_report()
        elif args.generate == 'weekly':
            scheduler.generate_weekly_summary()

    elif args.setup:
        # 设置调度方式
        if args.setup == 'cron':
            scheduler.setup_cron()
        elif args.setup == 'systemd':
            scheduler.setup_systemd()
        elif args.setup == 'apscheduler':
            print("APScheduler configuration:")
            print(json.dumps(scheduler.config['scheduler'], indent=2))

    elif args.run:
        # 运行调度器守护进程
        scheduler.run_scheduler()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()