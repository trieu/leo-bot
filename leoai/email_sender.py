import asyncio

from typing import Any, Dict, Optional
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import aiosmtplib
from jinja2 import Environment, FileSystemLoader, select_autoescape
from datetime import datetime
import traceback
from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv(override=True)


class EmailSender:
    def __init__(
        self,
        smtp_host: str = os.getenv("SMTP_HOST", "smtp.gmail.com"),
        smtp_port: int = int(os.getenv("SMTP_PORT", 587)),
        smtp_user: str = os.getenv("SMTP_USER", ""),
        smtp_password: str = os.getenv("SMTP_PASSWORD", ""),
        use_tls: bool = os.getenv("USE_TLS", "true"),
        template_dir: str = "./resources/templates/email_templates",
    ):
        # --- SMTP Config ---
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.use_tls = use_tls

        # --- Jinja2 Template Environment ---
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html", "xml"])
        )

    # -------------------------------------------------------------------------
    # Email Sending Logic
    # -------------------------------------------------------------------------
    async def send(
        self,
        to_email: str,
        subject: str,
        template_name: str,
        context: Optional[Dict[str, Any]] = None,
        from_email: Optional[str] = None,
    ) -> bool:
        """Render a Jinja2 template and send it as an email via SMTP."""
        try:
            context = context or self._create_base_context()
            template = self.jinja_env.get_template(template_name)
            html_content = template.render(**context)
            plain_text = template.render(**context)  # fallback: same as HTML

            # --- Build the email ---
            msg = MIMEMultipart("alternative")
            msg["From"] = from_email or self.smtp_user
            msg["To"] = to_email
            msg["Subject"] = subject

            msg.attach(MIMEText(plain_text, "plain"))
            msg.attach(MIMEText(html_content, "html"))

            # Send using async SMTP client
            # --- Send with aiosmtplib ---
            if self.use_tls:
                # Port 587 (STARTTLS)
                await aiosmtplib.send(
                    msg,
                    hostname=self.smtp_host,
                    port=self.smtp_port,
                    start_tls=True,
                    username=self.smtp_user,
                    password=self.smtp_password,
                )
            else:
                # Port 465 (SSL)
                await aiosmtplib.send(
                    msg,
                    hostname=self.smtp_host,
                    port=self.smtp_port,
                    use_tls=True,
                    username=self.smtp_user,
                    password=self.smtp_password,
                )
            return True
        except Exception as e:
            print(f" ❌ Email send failed: {e}")
            print("------ STACK TRACE ------")
            traceback.print_exc()
            print("--------------------------")
            return False


