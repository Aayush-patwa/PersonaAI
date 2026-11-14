# reminder.py - Gmail reminder system for JARVIS
import os
import smtplib
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from database import get_due_reminders

load_dotenv()

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

# ─────────────────────────────────────────
# EMAIL BUILDER
# ─────────────────────────────────────────

def build_email_body(tasks: list) -> str:
    """Build a clean HTML email body for reminders."""

    today = datetime.date.today()

    overdue = []
    due_today = []
    due_tomorrow = []

    for task in tasks:
        if task.get('due_date'):
            try:
                due = datetime.date.fromisoformat(task['due_date'])
                days_left = (due - today).days
                if days_left < 0:
                    overdue.append((task, abs(days_left)))
                elif days_left == 0:
                    due_today.append(task)
                elif days_left == 1:
                    due_tomorrow.append(task)
            except:
                pass

    html = """
    <html>
    <body style="font-family: Arial, sans-serif; background: #0a0a1a; color: #ffffff; padding: 30px;">
        <div style="max-width: 600px; margin: 0 auto; background: #1a1a2e; border-radius: 15px; padding: 30px; border: 1px solid #00d4ff;">

            <h1 style="color: #00d4ff; text-align: center; margin-bottom: 5px;">
                🤖 JARVIS Task Reminder
            </h1>
            <p style="color: #888; text-align: center; margin-bottom: 30px;">
                {date}
            </p>
    """.format(date=datetime.datetime.now().strftime("%A, %B %d, %Y"))

    # Overdue tasks
    if overdue:
        html += """
            <div style="background: #2d0a0a; border-left: 4px solid #ff4444; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                <h2 style="color: #ff4444; margin: 0 0 10px 0;">⚠️ Overdue Tasks</h2>
        """
        for task, days_ago in overdue:
            html += f"""
                <div style="padding: 8px 0; border-bottom: 1px solid #3d1a1a;">
                    <strong style="color: #ff6666;">{task['title']}</strong>
                    <span style="color: #ff4444; float: right;">{days_ago} day(s) overdue</span>
                </div>
            """
        html += "</div>"

    # Due today
    if due_today:
        html += """
            <div style="background: #2d1a00; border-left: 4px solid #ff9900; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                <h2 style="color: #ff9900; margin: 0 0 10px 0;">🔥 Due Today</h2>
        """
        for task in due_today:
            html += f"""
                <div style="padding: 8px 0; border-bottom: 1px solid #3d2a00;">
                    <strong style="color: #ffbb44;">{task['title']}</strong>
                    <span style="color: #ff9900; float: right;">Due Today!</span>
                </div>
            """
        html += "</div>"

    # Due tomorrow
    if due_tomorrow:
        html += """
            <div style="background: #0a1a2d; border-left: 4px solid #00d4ff; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                <h2 style="color: #00d4ff; margin: 0 0 10px 0;">📅 Due Tomorrow</h2>
        """
        for task in due_tomorrow:
            html += f"""
                <div style="padding: 8px 0; border-bottom: 1px solid #1a2a3d;">
                    <strong style="color: #44ccff;">{task['title']}</strong>
                    <span style="color: #00d4ff; float: right;">Tomorrow</span>
                </div>
            """
        html += "</div>"

    html += """
            <p style="color: #555; text-align: center; margin-top: 20px; font-size: 12px;">
                Sent by JARVIS — Your Personal AI Assistant
            </p>
        </div>
    </body>
    </html>
    """

    return html

# ─────────────────────────────────────────
# SEND EMAIL
# ─────────────────────────────────────────

def send_reminder_email(tasks: list) -> bool:
    """Send reminder email via Gmail."""
    try:
        if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
            print("❌ Gmail credentials not set in .env file")
            return False

        if not tasks:
            print("✅ No tasks due — no reminder needed")
            return True

        # Build email
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"🤖 JARVIS Reminder — {len(tasks)} task(s) need attention"
        msg['From'] = GMAIL_ADDRESS
        msg['To'] = GMAIL_ADDRESS

        html_body = build_email_body(tasks)
        msg.attach(MIMEText(html_body, 'html'))

        # Send via Gmail SMTP
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, GMAIL_ADDRESS, msg.as_string())

        print(f"✅ Reminder email sent for {len(tasks)} task(s)")
        return True

    except smtplib.SMTPAuthenticationError:
        print("❌ Gmail authentication failed. Check your App Password in .env")
        return False
    except Exception as e:
        print(f"❌ Error sending email: {str(e)}")
        return False

# ─────────────────────────────────────────
# CHECK AND SEND REMINDERS
# ─────────────────────────────────────────

def check_and_send_reminders():
    """Check for due tasks and send reminder email."""
    print(f"🔔 Checking reminders at {datetime.datetime.now().strftime('%H:%M')}...")
    tasks = get_due_reminders()

    if tasks:
        print(f"📋 Found {len(tasks)} task(s) due soon")
        send_reminder_email(tasks)
    else:
        print("✅ No tasks due soon")

# ─────────────────────────────────────────
# TEST EMAIL
# ─────────────────────────────────────────

def send_test_email() -> bool:
    """Send a test email to verify Gmail setup."""
    try:
        if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
            return False

        msg = MIMEMultipart('alternative')
        msg['Subject'] = "🤖 JARVIS — Test Email"
        msg['From'] = GMAIL_ADDRESS
        msg['To'] = GMAIL_ADDRESS

        html = """
        <html>
        <body style="font-family: Arial, sans-serif; background: #0a0a1a; color: #ffffff; padding: 30px;">
            <div style="max-width: 600px; margin: 0 auto; background: #1a1a2e; border-radius: 15px; padding: 30px; border: 1px solid #00d4ff;">
                <h1 style="color: #00d4ff; text-align: center;">🤖 JARVIS is Online</h1>
                <p style="color: #aaa; text-align: center;">
                    Your Gmail reminders are working perfectly.<br>
                    You will receive task reminders at 9:00 AM daily.
                </p>
            </div>
        </body>
        </html>
        """

        msg.attach(MIMEText(html, 'html'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, GMAIL_ADDRESS, msg.as_string())

        return True

    except Exception as e:
        print(f"Test email error: {e}")
        return False

if __name__ == '__main__':
    print("Testing reminder system...")
    check_and_send_reminders()