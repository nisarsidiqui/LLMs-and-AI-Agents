import gradio as gr
import os
import json
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import base64
from datetime import datetime, timedelta

# Gmail API scope
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Instead of reading from file, get credentials from environment variables
CLIENT_CONFIG = {
    "installed": {
        "client_id": os.environ.get('GOOGLE_CLIENT_ID'),
        "client_secret": os.environ.get('GOOGLE_CLIENT_SECRET'),
        "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
}

# Keywords for filtering emails
KEYWORDS = {
    'job_related': [
        'job opportunity', 'position', 'career', 'recruitment', 'hiring',
        'interview', 'resume', 'CV', 'application', 'job posting',
        'employment', 'role', 'vacancy', 'opening'
    ],
    'personal': [
        'personal', 'private', 'confidential', 'family', 'friend',
        'social', 'invitation', 'gathering', 'meetup'
    ]
}

def get_gmail_service(state_dict):
    """Creates Gmail API service"""
    creds = None
    
    # Check if token exists in state
    if 'token' in state_dict:
        creds = Credentials.from_authorized_user_info(state_dict['token'], SCOPES)

    # If credentials are invalid or don't exist
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_config(CLIENT_CONFIG, SCOPES)
            creds = flow.run_local_server(port=0)
            
            # Save credentials in state
            state_dict['token'] = {
                'token': creds.token,
                'refresh_token': creds.refresh_token,
                'token_uri': creds.token_uri,
                'client_id': creds.client_id,
                'client_secret': creds.client_secret,
                'scopes': creds.scopes
            }

    return build('gmail', 'v1', credentials=creds)

def get_email_content(service, msg_id):
    """Retrieves email content and metadata"""
    try:
        message = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
        
        headers = message['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
        from_email = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'No Sender')
        date = next((h['value'] for h in headers if h['name'].lower() == 'date'), 'No Date')
        
        # Get email body
        if 'parts' in message['payload']:
            parts = message['payload']['parts']
            data = parts[0]['body'].get('data', '')
        else:
            data = message['payload']['body'].get('data', '')
            
        if data:
            text = base64.urlsafe_b64decode(data).decode('utf-8')
        else:
            text = "No content available"
            
        # Check for attachments
        attachments = []
        if 'parts' in message['payload']:
            for part in message['payload']['parts']:
                if 'filename' in part and part['filename']:
                    attachments.append(part['filename'])
        
        return {
            'subject': subject,
            'from': from_email,
            'date': date,
            'content': text,
            'attachments': attachments
        }
    except Exception as e:
        return f"Error retrieving email: {str(e)}"

def classify_email(email_data):
    """Classifies email based on content and attachments"""
    text = f"{email_data['subject']} {email_data['content']}".lower()
    
    # Check attachments for CV/Resume
    has_cv = any(
        att.lower().endswith(('.pdf', '.doc', '.docx')) or 
        any(kw in att.lower() for kw in ['cv', 'resume']) 
        for att in email_data['attachments']
    )
    
    # Check content for keywords
    is_job_related = has_cv or any(kw.lower() in text for kw in KEYWORDS['job_related'])
    is_personal = any(kw.lower() in text for kw in KEYWORDS['personal'])
    
    return {
        'job_related': is_job_related,
        'personal': is_personal,
        'has_cv': has_cv
    }

def fetch_emails(days_back, include_job=True, include_personal=True, progress=gr.Progress()):
    """Main function to fetch and filter emails"""
    state_dict = {}
    try:
        service = get_gmail_service(state_dict)
        
        # Search for recent emails
        query = f'after:{int((datetime.now() - timedelta(days=int(days_back))).timestamp())}'
        results = service.users().messages().list(userId='me', q=query, maxResults=100).execute()
        messages = results.get('messages', [])
        
        if not messages:
            return "No emails found in the specified time range."
        
        filtered_emails = []
        
        # Process emails with progress tracking
        for i, message in enumerate(messages):
            progress(i/len(messages), desc="Processing emails...")
            email_data = get_email_content(service, message['id'])
            if isinstance(email_data, str):  # Error message
                continue
                
            classification = classify_email(email_data)
            email_data.update(classification)
            
            if ((include_job and classification['job_related']) or 
                (include_personal and classification['personal'])):
                filtered_emails.append(email_data)
        
        # Format output
        output = f"Found {len(filtered_emails)} matching emails\n\n"
        for email in filtered_emails:
            output += f"📧 {email['subject']}\n"
            output += f"From: {email['from']}\n"
            output += f"Date: {email['date']}\n"
            
            tags = []
            if email['job_related']:
                tags.append("🎯 Job Related")
            if email['personal']:
                tags.append("👤 Personal")
            if email['has_cv']:
                tags.append("📎 Has CV/Resume")
            
            output += f"Tags: {', '.join(tags)}\n"
            
            if email['attachments']:
                output += "Attachments:\n"
                for att in email['attachments']:
                    output += f"- {att}\n"
            
            output += "\nContent Preview:\n"
            preview = email['content'][:500] + "..." if len(email['content']) > 500 else email['content']
            output += f"{preview}\n"
            output += "-" * 80 + "\n\n"
        
        return output
        
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Email Filter") as demo:
        gr.Markdown("# 📧 Smart Email Filter")
        gr.Markdown("Connect to your Gmail account to filter important emails")
        
        with gr.Row():
            days_back = gr.Slider(
                minimum=1, 
                maximum=30, 
                value=7, 
                step=1, 
                label="Days to look back"
            )
            include_job = gr.Checkbox(
                value=True, 
                label="Include Job Related Emails"
            )
            include_personal = gr.Checkbox(
                value=True, 
                label="Include Personal Emails"
            )
        
        fetch_button = gr.Button("Connect and Fetch Emails")
        output = gr.Textbox(
            label="Results", 
            lines=20,
            show_copy_button=True
        )
        
        fetch_button.click(
            fn=fetch_emails,
            inputs=[days_back, include_job, include_personal],
            outputs=output
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()