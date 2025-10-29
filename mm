import praw
import time
import csv
import random
import json
import os
import unicodedata
from prawcore.exceptions import RequestException, ResponseException, ServerError, Forbidden
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from dotenv import load_dotenv

# LangChain imports
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Please set OPENROUTER_API_KEY in .env or env variables")

# CONFIG
SUBREDDIT = "AskReddit"
DAYS = 30
TOP_PERCENT = 0.01
CSV_FILENAME = f"top_1_percent_commenters_{SUBREDDIT}.csv"
COMMENT_LOG = "commented_posts.json"
MAX_COMMENTS_PER_DAY = 10  # Safety limit

# AUTHENTICATION - Consider moving these to environment variables
ac = {
    "client_id": "qNKQjHDzMz5nV73GCGVJRg",
    "client_secret": "EuLhNujMG14QQPHN9Y3AqgWNOTtZw",
    "username": "simplicityesqueED",
    "password": "kBMm3053mY2RL045"
}

reddit = praw.Reddit(
    client_id=ac["client_id"],
    client_secret=ac["client_secret"],
    username=ac["username"],
    password=ac["password"],
    user_agent=f"script:multi-account-bot:v1 (by u/{ac['username']})"
)

# === LangChain Setup ===
def create_langchain_client():
    """Initialize LangChain with OpenRouter"""
    return ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=OPENROUTER_API_KEY,
        model_name="openai/gpt-4o-mini",
        temperature=0.7,
        max_tokens=200
    )

chat_model = create_langchain_client()

# === Enhanced Prompt Templates ===
def create_comment_prompt():
    """Create structured prompt for generating Reddit comments"""
    
    response_schemas = [
        ResponseSchema(name="comment", description="The actual comment text to post on Reddit"),
        ResponseSchema(name="reasoning", description="Brief reasoning why this comment is appropriate"),
        ResponseSchema(name="safety_score", description="Safety score from 1-10 (10 being safest)")
    ]
    
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
    system_template = """You are a helpful, engaging Reddit user participating in r/AskReddit discussions.

CRITICAL RULES:
1. Generate ONLY natural, human-like comments
2. NEVER reveal you are an AI
3. ALWAYS be respectful and follow Reddit etiquette
4. Avoid controversial, offensive, or sensitive topics
5. Keep comments concise (1-3 sentences)
6. Add value to the discussion - don't just agree
7. Use appropriate humor when suitable
8. NEVER generate markdown, links, or special formatting

Safety Guidelines:
- Score below 7/10 safety: DO NOT post
- Avoid medical, financial, or legal advice
- No politics, religion, or sensitive personal topics
- Be positive and constructive"""

    human_template = """
Post Title: {post_title}
Post Content: {post_content}
Existing Comments:
{existing_comments}

Generate a thoughtful, engaging comment that adds value to this discussion."""

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])
    
    return chat_prompt, output_parser

def create_safety_check_prompt():
    """Create prompt for safety checking comments"""
    safety_system = """Analyze this Reddit comment for safety and appropriateness.
    
Evaluate on:
1. Toxicity (insults, hate speech)
2. Controversy (politics, religion)
3. Personal advice (medical, legal, financial)
4. Spam characteristics
5. Overall appropriateness for r/AskReddit

Return ONLY a score from 1-10 where:
10: Perfectly safe and appropriate
7-9: Mostly safe, minor concerns
4-6: Questionable, needs review
1-3: Dangerous or inappropriate"""

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(safety_system),
        HumanMessagePromptTemplate.from_template("Comment to analyze: {comment}")
    ])

# === UTILITY: Retry Decorator ===
def retry_on_fail(func, max_retries=5, delay=5):
    for attempt in range(max_retries):
        try:
            return func()
        except (RequestException, ResponseException, ServerError, Forbidden) as e:
            print(f"[Retry {attempt+1}] Error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
    raise Exception(f"Failed after {max_retries} retries.")

# === Step 1: Get Submissions from Last 30 Days ===
def get_recent_submissions(subreddit_name, days):
    subreddit = reddit.subreddit(subreddit_name)
    start_time = datetime.utcnow() - timedelta(days=days)
    start_timestamp = int(start_time.timestamp())
    
    print(f"Fetching submissions from r/{subreddit_name} (last {days} days)...")
    submissions = []
    for submission in retry_on_fail(lambda: list(subreddit.new(limit=1000))):
        if submission.created_utc >= start_timestamp:
            submissions.append(submission)
    
    print(f"Fetched {len(submissions)} submissions.")
    return submissions

# === Step 2: Get Commenters and Data ===
def get_commenters_and_data(submissions):
    user_comments = defaultdict(int)
    submission_data = []
    
    for submission in submissions:
        try:
            retry_on_fail(lambda: submission.comments.replace_more(limit=0))
            all_comments = submission.comments.list()
            
            for comment in all_comments:
                if comment.author and comment.author.name != "AutoModerator":
                    user_comments[comment.author.name] += 1
            
            submission_data.append({
                "id": submission.id,
                "submission": submission,
                "comments": all_comments
            })
        except Exception as e:
            print(f"Skipping submission due to error: {e}")
    
    print(f"Collected {len(user_comments)} unique commenters.")
    return user_comments, submission_data

# === Step 3: Get Top 1% Commenters ===
def get_top_percent_users(user_counts, percent):
    total_users = len(user_counts)
    top_n = max(1, int(total_users * percent))
    top_users = Counter(user_counts).most_common(top_n)
    print(f"Top {percent*100:.0f}% = {len(top_users)} users.")
    return top_users

# === Step 4: Save to CSV ===
def save_to_csv(data, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Username", "Comment Count"])
        for username, count in data:
            writer.writerow([username, count])
    print(f"Saved results to {filename}")

# === Enhanced Comment Generation with LangChain ===
def sanitize_text(text: str) -> str:
    """Normalize and clean text for Reddit"""
    text = unicodedata.normalize("NFKD", text)
    text = text.replace('\u2011', '-')
    text = text.replace('\u00A0', ' ')
    text = text.replace('"', "'").replace('"', "'")
    return text.encode('ascii', 'ignore').decode('ascii')

def safety_check(comment_text):
    """Check if comment is safe to post"""
    safety_prompt = create_safety_check_prompt()
    messages = safety_prompt.format_prompt(comment=comment_text).to_messages()
    
    try:
        response = chat_model(messages)
        safety_score = int(response.content.strip())
        return safety_score >= 7  # Only post if score is 7 or higher
    except:
        return False  # Default to unsafe if check fails

def generate_comment_with_langchain(post, comments):
    """Generate comment using LangChain with safety checks"""
    
    chat_prompt, output_parser = create_comment_prompt()
    
    # Prepare context
    post_content = post.selftext[:800] if post.selftext else "[No text content]"
    existing_comments = "\n---\n".join([
        f"User: {c.author.name if c.author else 'deleted'}\nComment: {c.body[:200]}"
        for c in comments[:3]  # Use fewer comments for better context
    ])
    
    messages = chat_prompt.format_prompt(
        post_title=post.title,
        post_content=post_content,
        existing_comments=existing_comments
    ).to_messages()
    
    response = chat_model(messages)
    
    try:
        parsed_output = output_parser.parse(response.content)
        comment_text = parsed_output["comment"]
        safety_score = parsed_output.get("safety_score", 5)
        
        # Additional safety check
        if safety_score >= 7 and safety_check(comment_text):
            return sanitize_text(comment_text)
        else:
            print(f"Comment failed safety check (score: {safety_score})")
            return None
            
    except Exception as e:
        print(f"Error parsing comment: {e}")
        return None

# === Step 6: Enhanced Commenting with Safety ===
def load_commented_log():
    try:
        with open(COMMENT_LOG, "r") as f:
            data = json.load(f)
            return set(data.get("commented_ids", []))
    except:
        return set()

def save_commented_log(commented_ids):
    data = {
        "commented_ids": list(commented_ids),
        "last_updated": datetime.utcnow().isoformat()
    }
    with open(COMMENT_LOG, "w") as f:
        json.dump(data, f, indent=2)

def get_daily_comment_count():
    """Check how many comments we've made today"""
    try:
        with open(COMMENT_LOG, "r") as f:
            data = json.load(f)
            last_updated = data.get("last_updated")
            if last_updated:
                last_date = datetime.fromisoformat(last_updated).date()
                if last_date == datetime.utcnow().date():
                    return data.get("daily_count", 0)
    except:
        pass
    return 0

def update_daily_count(count):
    """Update daily comment count"""
    try:
        with open(COMMENT_LOG, "r") as f:
            data = json.load(f)
    except:
        data = {}
    
    data["daily_count"] = count
    data["last_updated"] = datetime.utcnow().isoformat()
    
    with open(COMMENT_LOG, "w") as f:
        json.dump(data, f, indent=2)

def comment_on_posts(submission_data, target_count):
    commented_ids = load_commented_log()
    daily_count = get_daily_comment_count()
    count = 0
    
    # Respect daily limits
    if daily_count >= MAX_COMMENTS_PER_DAY:
        print(f"Daily limit reached ({MAX_COMMENTS_PER_DAY} comments). Skipping.")
        return
    
    random.shuffle(submission_data)
    
    for entry in submission_data:
        if count >= target_count or (daily_count + count) >= MAX_COMMENTS_PER_DAY:
            break
            
        sid = entry["id"]
        if sid in commented_ids:
            continue
            
        submission = entry["submission"]
        comments = entry["comments"]
        
        if not comments:
            continue
        
        try:
            # Skip if post is too old or has too many comments
            post_age = datetime.utcnow() - datetime.fromtimestamp(submission.created_utc)
            if post_age.days > 7 or len(comments) > 500:
                continue
            
            text = generate_comment_with_langchain(submission, comments)
            
            if text and len(text) > 10:  # Basic validation
                submission.reply(text)
                print(f"Commented on https://reddit.com/comments/{submission.id}")
                print(f"Title: {submission.title[:100]}...")
                print(f"Comment: {text}\n")
                
                count += 1
                commented_ids.add(sid)
                save_commented_log(commented_ids)
                
                # Be respectful with delays
                delay = random.randint(60, 180)  # 1-3 minutes
                time.sleep(delay)
            else:
                print("Skipping post due to comment generation issues")
                
        except Exception as e:
            print(f"Error commenting: {e}")
            time.sleep(30)  # Shorter delay on error
    
    # Update daily count
    new_daily_count = daily_count + count
    update_daily_count(new_daily_count)
    print(f"Posted {count} comments today. Daily total: {new_daily_count}")

# === MAIN ===
if __name__ == "__main__":
    print("Starting enhanced Reddit bot with LangChain...")
    
    # Get submissions and analyze
    submissions = get_recent_submissions(SUBREDDIT, DAYS)
    user_counts, submission_data = get_commenters_and_data(submissions)
    top_users = get_top_percent_users(user_counts, TOP_PERCENT)
    
    for username, count in top_users[:10]:  # Show top 10
        print(f"u/{username}: {count} comments")
    
    save_to_csv(top_users, CSV_FILENAME)
    
    # Match top commenter activity (with limits)
    if top_users:
        top_comment_count = min(top_users[0][1], MAX_COMMENTS_PER_DAY)
        print(f"Aiming to comment on ~{top_comment_count} posts (with daily limits).")
        comment_on_posts(submission_data, top_comment_count)
    else:
        print("No top commenters found.")
