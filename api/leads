"""
api/leads.py  —  Vercel Serverless Function
Captures email, stores lead, sends report email via Resend.

Set these in Vercel dashboard → Settings → Environment Variables:
  RESEND_API_KEY   =  re_xxxxxxxxxxxx
  FROM_EMAIL       =  hello@smartagentx.ai
  SITE_URL         =  https://pulse.smartagentx.ai
"""

import json, os, asyncio
import httpx
from http.server import BaseHTTPRequestHandler

RESEND_KEY  = os.environ.get("RESEND_API_KEY", "")
FROM_EMAIL  = os.environ.get("FROM_EMAIL", "hello@smartagentx.ai")
SITE_URL    = os.environ.get("SITE_URL", "https://pulse.smartagentx.ai")

# In production, replace this with Supabase or a simple JSON file on Vercel KV
_leads = []


async def send_email(to: str, name: str, domain: str, score: int,
                     tier: str, color: str, gaps: list, report_id: str):
    if not RESEND_KEY:
        print(f"[EMAIL] Would send to {to} | Score: {score} | Tier: {tier}")
        return

    gaps_html = "".join(f"<li style='margin:6px 0;color:#3d3d3a'>{g}</li>" for g in gaps[:3])

    html = f"""
<div style="font-family:system-ui,sans-serif;max-width:580px;margin:0 auto;background:#faf9f6">
  <div style="padding:24px 40px;border-bottom:1px solid #e4e0d6">
    <span style="font-family:monospace;font-size:13px;font-weight:700;color:#1a1a18">
      smart<span style="color:#534AB7">agent</span>x
    </span>
  </div>
  <div style="padding:40px">
    <p style="font-size:16px;margin:0 0 20px">Hi {name},</p>
    <p style="font-size:15px;color:#3d3d3a;margin:0 0 28px;line-height:1.6">
      Here is your full Agentic Pulse Score report for <strong>{domain}</strong>.
    </p>

    <div style="background:#f0ede6;border-radius:14px;padding:32px;text-align:center;margin:0 0 28px">
      <div style="font-size:80px;font-weight:700;color:{color};line-height:1">{score}</div>
      <div style="font-size:13px;color:#7a7a75;margin:6px 0 14px">out of 100</div>
      <div style="display:inline-block;background:{color};color:white;padding:4px 16px;
           border-radius:20px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em">
        {tier.replace("-"," ")}
      </div>
    </div>

    <p style="font-size:14px;font-weight:600;color:#1a1a18;margin:0 0 10px">
      Your top 3 gaps to fix:
    </p>
    <ol style="padding-left:20px;margin:0 0 28px;line-height:1.8;font-size:14px">
      {gaps_html}
    </ol>

    <div style="background:#1a1a18;border-radius:12px;padding:24px;text-align:center;margin:0 0 24px">
      <p style="color:white;font-size:15px;margin:0 0 16px;line-height:1.6">
        Ready to fix your score and capture AI-driven revenue?
      </p>
      <a href="{SITE_URL}?report={report_id}"
         style="display:inline-block;background:white;color:#1a1a18;padding:13px 26px;
                border-radius:8px;text-decoration:none;font-weight:700;font-size:14px">
        Get AI-ready → from $497
      </a>
    </div>

    <p style="font-size:13px;color:#7a7a75;line-height:1.6">
      Questions? Reply to this email — I read every response personally.
    </p>
  </div>
  <div style="padding:20px 40px;border-top:1px solid #e4e0d6;text-align:center">
    <p style="font-size:11px;color:#7a7a75;font-family:monospace;margin:0">
      SmartAgentX · {SITE_URL}<br>
      <a href="#" style="color:#7a7a75">Unsubscribe</a>
    </p>
  </div>
</div>"""

    async with httpx.AsyncClient() as client:
        await client.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_KEY}",
                     "Content-Type": "application/json"},
            json={
                "from": f"SmartAgentX <{FROM_EMAIL}>",
                "to": to,
                "subject": f"Your Agentic Pulse Score: {score}/100 — {domain}",
                "html": html,
            },
            timeout=10,
        )


class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)
        try:
            data = json.loads(body)
            email = data.get("email", "").strip()
            if not email or "@" not in email:
                raise ValueError("Valid email required")
        except Exception as e:
            self._json({"error": str(e)}, 400)
            return

        lead = {
            "email":      email,
            "name":       data.get("name", ""),
            "domain":     data.get("domain", ""),
            "score":      data.get("score", 0),
            "tier":       data.get("tier", ""),
            "report_id":  data.get("report_id", ""),
        }
        _leads.append(lead)

        try:
            asyncio.run(send_email(
                to=email,
                name=lead["name"] or "there",
                domain=lead["domain"],
                score=lead["score"],
                tier=lead["tier"],
                color=data.get("color", "#534AB7"),
                gaps=data.get("top_gaps", []),
                report_id=lead["report_id"],
            ))
        except Exception as e:
            print(f"Email send error: {e}")

        self._json({"success": True,
                    "message": "Report sent — check your inbox!"})

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, *args):
        pass
