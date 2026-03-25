"""
Sentiment Analyzer Module
Analyzes messages based on rules defined in sentiment.md
"""

import re
import os


class SentimentAnalyzer:
    """Analyzes Discord messages for negative sentiment based on predefined rules."""

    def __init__(self, sentiment_rules_file='sentiment.md', manual_examples_file='sentimentManual.md'):
        """
        Initialize sentiment analyzer with rules from sentiment.md and examples from sentimentManual.md

        Args:
            sentiment_rules_file: Path to sentiment rules markdown file
            manual_examples_file: Path to manual negative examples file
        """
        self.sentiment_context = self._load_sentiment_context(sentiment_rules_file)
        self.manual_examples_loaded = self._check_manual_examples(manual_examples_file)

        # -----------------------------------------------------------------------
        # HARD EXCLUSION PATTERNS — if any of these match, message is NEUTRAL
        # regardless of other signals. Derived from Document 3 analysis
        # (messages wrongly flagged as negative).
        # -----------------------------------------------------------------------

        # 1. Positive emojis — messages with these are almost never complaints
        self.positive_emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F'
            r'\U0001F680-\U0001F6FF'
            r'\U00002702-\U000027B0'
            r'\U0001F300-\U0001F5FF'
            r'\U0001F1E0-\U0001F1FF'
            r'\u2600-\u26FF'
            r'\u2700-\u27BF]',
            re.UNICODE
        )

        # 2. New-user / onboarding confusion — joining, navigation, group discovery
        # These are NOT complaints; they are orientation questions from new joiners
        self.onboarding_confusion_patterns = [
            r'\b(what is this (group|channel|server)|what (group|channel) is this)\b',
            r'\b(is this the (right|correct) (group|channel|batch))\b',
            r'\b(how (do|can) i (join|access) (the )?(group|channel|batch|server))\b',
            r'\b(how to join|how to access)\b.*\b(group|channel|batch|server)\b',
            r'\b(i (am|m) (new|a new joinee|just joined)|just joined|newly joined)\b',
            r'\b(not (sure|know) (which|what) (group|channel))\b',
            r'\b(wrong (group|channel|batch)|galat (group|channel))\b',
            r'\binvite (code|link)\b',
            r'\b(get[ -]?access (link|channel))\b',
            r'\b(how to (use|navigate|find|open) (discord|this app|this platform))\b',
            r'\b(discord (is|seems) (confusing|new|hard|difficult))\b',
            r'\b(new to discord|not used to (this|discord))\b',
            r'\b(i (don\'?t|dont) (see|find) (any )?(channel|group|voice channel|message))\b',
            r'\b(is there (any other|another|a different) (group|channel|batch))\b',
            r'\b(am i in the (right|correct) (group|channel|batch|place))\b',
            r'\b(added to (any|the) (group|channel|batch)|not added to)\b',
            r'\b(can (someone|anyone) (add|send|share) (me )?(the )?(link|invite|channel))\b',
            r'\b(can (you|someone) help me (add|join|get access))\b',
            r'\b(please (add|send|share) (me |the )?(link|invite|channel|access))\b',
        ]

        # 3. Class schedule inquiries — simple yes/no questions about class timing
        # These look negative ("no class?", "class cancelled?") but are neutral
        self.schedule_inquiry_patterns = [
            r'^\s*(is there|do we have|don\'?t we have|are we having)\b.*\b(class|session|lecture)\b.*\??\s*$',
            r'^\s*(class (today|tomorrow|tonight|this week|on monday|on friday|on saturday|on sunday))\s*\??\s*$',
            r'^\s*(no class (today|tomorrow|tonight|this week))\s*\??\s*$',
            r'^\s*(today (no class|class cancelled|class not there))\s*\??\s*$',
            r'\b(class (today|tomorrow|tonight)\?+)\b',
            r'\b(when (is|are|will) (the )?(next )?(class|session|lecture))\b',
            r'\b(class (timing|time|schedule|link))\b.*\?',
            r'\b(is (the )?(class|session|lecture) (cancelled|rescheduled|postponed))\b',
            r'\b(do we have (a )?(break|off|holiday) (today|tomorrow|this week))\b',
            r'^\s*(don\'?t we have (a )?class (today|tomorrow))\s*\??\s*$',
        ]

        # 4. Instructor/support team redirect templates
        # These are REPLIES to complaints, not themselves complaints
        self.support_redirect_templates = [
            r'\b(please (raise|create|submit|log) (a )?(ticket|support request|request))\b',
            r'\b(reach out to (our |the )?(support team|support|team))\b',
            r'\b(contact (our |the )?(support team|support))\b',
            r'\bhttps?://shorturl\.at/\w+\b',
            r'\b(chatbot is available 24.?7)\b',
            r'\b(want a callback.{0,30}request one between)\b',
            r'\b(we\'?re (here|always here) to help)\b',
            r'\b(support team (will|shall) (get back|reach out|contact|call) (you|to you))\b',
            r'\b(apolog(y|ies|ize) for the (inconvenience|delay|confusion))\b',
            r'\b(team is (working on|looking into|checking|investigating) (this|it))\b',
            r'\b(escalat(ed|ing) (this|your concern|the issue) to (the )?(concerned|relevant) team)\b',
        ]

        # 5. Instructor announcements — upcoming class previews, session agendas
        self.instructor_announcement_patterns = [
            r'\b(in (the |our )?(upcoming|next|today\'?s|tonight\'?s) (lecture|session|class))\b',
            r'\b(we\'?ll (cover|explore|dive into|tackle|discuss|learn))\b',
            r'\b(see you (in (the )?class|there|all|at|tomorrow|tonight))\b',
            r'\b(looking forward to (our |the |seeing|meeting))\b',
            r'\b(join (us|the class|me|the session) (today|tomorrow|tonight|at|for))\b',
            r'\b(please (plan to|do) (attend|join))\b',
            r'\b(don\'?t miss (it|today\'?s|the))\b',
            r'\b(by the end (of (this|the) (session|lecture|class)))\b',
            r'\b(get ready for)\b',
            r'\b(let\'?s (explore|dive|get started|solve|discuss|understand))\b',
            r'\b(sharing (all )?(the )?(topics|agenda|content|schedule))\b',
        ]

        # 6. Peer helping and collaborative messages
        self.collaborative_patterns = [
            r'\b(hope this helps|hope that helps|this might help|found this useful)\b',
            r'\b(sharing (this|a link|the link|notes|resource|solution))\b',
            r'\b(check (this|it) out|refer to this|you can refer)\b',
            r'\b(anyone (know|have|solved|tried|done))\b',
            r'\b(does anyone (have|know|see|understand))\b',
            r'\b(can (someone|anyone) (share|send|help|explain|tell))\b',
        ]

        # 7. Coding/technical learning questions (not complaints)
        self.coding_learning_patterns = [
            r'\b(help (me )?(with|understand|solve|debug|fix|explain))\b.*\b(code|function|problem|question|query|algorithm|error|exception|bug)\b',
            r'\b(what is (wrong|incorrect|the issue) (with|in) (my|the) code)\b',
            r'\b(can (anyone|someone) (explain|help|clarify))\b.*\b(concept|topic|logic|approach|solution)\b',
            r'\b(not (getting|understanding|able to understand) (the )?(concept|logic|approach|output|result))\b',
            r'\b(test (case|cases) (failing|not passing|are wrong))\b',
            r'\b(my (code|solution|query|approach) (is|seems) (correct|right) but)\b',
            r'\b(getting (error|exception|wrong output|wrong answer))\b.*\b(code|function|query|program)\b',
        ]

        # 8. Positive/neutral standalone phrases
        self.positive_neutral_phrases = [
            r'\b(happy learning|keep learning|good luck|all the best|best of luck)\b',
            r'\b(thank(s| you)|great|awesome|wonderful|excellent|perfect|super|well done|congrats)\b',
            r'\b(happy new year|happy holi|happy diwali|festival|celebration)\b',
            r'\b(revising|revision|practicing|practice|completing|solved it|got it|issue resolved|resolved)\b',
            r'\b(no worries|don\'?t worry|it\'?s ok|its ok|no problem|that\'?s fine)\b',
        ]

        # -----------------------------------------------------------------------
        # NEGATIVE SIGNAL TIERS
        # A message needs to hit at least one of these to be flagged negative.
        # Hard exclusions above override all of these.
        # -----------------------------------------------------------------------

        # VERY STRONG — alone sufficient (derived from Doc 4 true negatives)
        self.very_strong_negative = [
            # Explicit refund / quit intent
            r'\b(refund|paisa wapas|paise vapas|money back|want.*refund|request.*refund)\b',
            r'\b(quit|leaving|discontinue|cancel.*admission|cancel.*course)\b.*\b(course|program|batch|scaler)\b',
            r'\b(chhod|chod|chodna|chhodna)\b.*\b(course|program|class|scaler)\b',

            # Legal / escalation language
            r'\b(legal (action|standpoint|notice)|compelled to (take|consider) action|from.*legal.*standpoint)\b',
            r'\b(consumer (forum|court)|file.*complaint.*against)\b',
            r'\b(escalate.*management|escalate.*senior|escalate.*founder)\b',

            # Money wasted
            r'\b(wasted (my )?(money|3 (lakh|lac)|time and money)|waste of money|waste of.*lakhs)\b',
            r'\b(paying.*but (not|nothing)|paid.*and (not|nothing))\b.*\b(getting|received|helping)\b',
            r'\b(fake (promises|marketing|jobs)|false promises|cheated|fraud|scam)\b.*\b(scaler|course|program)\b',
            r'\b(paisa.*barbaad|barbad.*paisa|paise.*waste)\b',

            # Placement failure rage
            r'\b(placement (cell|team) (is|are) (useless|pathetic|fake|joke|not working|doing nothing))\b',
            r'\b(not getting (any )?(calls|interviews|response) from (scaler|placement))\b',
            r'\b(applied.*times.*(no|not any|zero) (response|call|interview))\b',
            r'\b(zero (placements|jobs|opportunities|response))\b',
            r'\b(3 (lakh|lac).*wasted|wasted.*3 (lakh|lac))\b',

            # Explicit course rage
            r'\b(worst (course|instructor|experience|batch|support)|pathetic (course|support|experience|instructor))\b',
            r'\b(never (recommend|suggesting) scaler)\b',
            r'\b(biggest (mistake|regret) (of my life|joining scaler|enrolling))\b',

            # Hindi explicit anger
            r'\b(gussa|pareshan|preshan|thak gaya|thak gayi|tang aa gaya|tang aa gayi)\b.*\b(scaler|support|course)\b',
            r'\b(jhooth|jhoot|dhoka|dhokha)\b.*\b(scaler|course|promise|marketing)\b',
            r'\b(ghatiya|bakwas|bekar|faltu)\b.*\b(course|support|scaler|instructor|class)\b',
        ]

        # STRONG — alone sufficient (explicit, unambiguous complaints)
        self.strong_negative = [
            # Repeated failed support attempts
            r'\b(tried (multiple|many|several|3|4|5|\d+) times.{0,30}(no response|not (resolved|fixed|answered|working)))\b',
            r'\b(raised.*(ticket|complaint|request).*(multiple|many|several|\d+) times)\b',
            r'\b(reached out.*(multiple|many|several|\d+) times.{0,40}(no|not any) (response|reply|update|action))\b',
            r'\b(still (not|no) (resolved|fixed|answered|addressed|response|update|progress)).{0,50}(days|weeks|since|long)\b',
            r'\b(waited (for )?(days|weeks|long|so long).{0,30}(no|not any) (response|reply|update|call))\b',

            # Explicit disappointment about a specific failure
            r'\b(regret (joining|enrolling|taking|decision to join))\b',
            r'\b(joining scaler was (a mistake|wrong|bad decision))\b',
            r'\b(disappointed (with|by|in) scaler)\b',
            r'\b(fed up (with|of) (scaler|this course|support|instructor))\b',
            r'\b(frustrated (with|by|about) (scaler|support|instructor|course|platform))\b',

            # Batch/schedule changed without notice
            r'\b(batch (changed|shifted|moved) (without|without any) (notice|information|communication|poll))\b',
            r'\b(curriculum (changed|modified) (without|without any) (notice|information|communication))\b',
            r'\b(changed (without|without any) notice)\b',

            # Explicit support failure
            r'\b(support (team|bot|chatbot) (not working|not responding|not helpful|useless|pathetic))\b',
            r'\b(no one (from )?(scaler )?(is )?(responding|replying|helping|available))\b',
            r'\b(no response from (scaler|support|TA|instructor|team))\b.*\b(days|weeks|long|since)\b',

            # Instructor change complaint (with frustration)
            r'\b(not (happy|ok|okay|comfortable|satisfied) with (instructor|teacher) change)\b',
            r'\b(why (is|are) (they|you) changing (the )?instructor)\b',

            # Platform breaking course progress
            r'\b(contest (disqualified|invalidated) (due to|because of) (platform|technical|system) (issue|error|bug))\b',
            r'\b(got disqualified.{0,30}(platform|technical|system) (issue|error|bug))\b',
            r'\b(lost.{0,15}(attempt|chance|score).{0,30}(platform|technical|system|bug|error))\b',

            # Certificate/placement promise broken
            r'\b(certificate (is|was) (fake|wrong|incorrect|not genuine|not verified|not recognized))\b',
            r'\b(promised.{0,30}(but|however|yet|and).{0,30}(not|never|nothing))\b',
            r'\b(sales (executive|person|team) (promised|said|told).{0,50}(not|never|nothing|false|lie))\b',
        ]

        # MODERATE — need 2+ to trigger (ambiguous, need corroboration)
        self.moderate_negative = [
            # Support failures (need corroboration)
            r'\b(unable to reach (support|TA|instructor))\b',
            r'\b(not (getting|receiving) (a )?(call ?back|response|reply))\b',
            r'\b(haven\'?t (received|gotten|got) (a |any )?(response|reply|call|update))\b',
            r'\b(ticket (raised|submitted) (but |and )(no |still no |nothing|not))\b',
            r'\b(waiting for (a )?(response|reply|update|call|callback))\b',
            r'\b(messaged.*personally.*still (waiting|no response))\b',
            r'\b(tried.*reach(ing)?.*but (no|not) (response|reply|answer))\b',
            r'\b(koi jawab nahi|reply nahi|kuch response nahi)\b',

            # Technical issues blocking course (not just navigation)
            r'\b(dashboard.*(lagging|stuck|not loading|problem|issue|error))\b',
            r'\b(platform.*(not working|broken|crashed|down|error))\b',
            r'\b(assignment.*(not submitting|not accepting|wrong answer|submission.*fail))\b',
            r'\b(login.*not working|not able to login)\b.*\b(scaler|dashboard|platform)\b',
            r'\b(recording.*(not available|missing|not uploaded|can\'?t (find|access|see)))\b',

            # Curriculum quality complaints
            r'\b(not covered in (class|lecture|session)).{0,40}(but|yet|however|still) (in )?(assignment|test|contest)\b',
            r'\b(assignment (questions|problems).{0,30}(not taught|not covered|out of syllabus))\b',
            r'\b(out of syllabus (question|topic|content))\b',

            # Batch concerns (specific)
            r'\b(cannot attend (live )?(class|session).{0,30}(time|timing|schedule|batch change))\b',
            r'\b(directly impacting (my )?(career|learning|progress))\b',
            r'\b(losing (my )?(PSP|streak|score|progress) (because|due to|because of))\b',

            # Hindi moderate negatives
            r'\b(nahi mila|nahi aaya|koi pick nahi|call back nahi)\b',
            r'\b(support nahi mil raha|koi help nahi)\b',

            # Agreement with complaints (with complaint word)
            r'\b(same (here|issue|problem|thing|happening) with me)\b',
            r'\b(i (also|too) (am|have been|was) facing (the )?same)\b',
        ]

        # -----------------------------------------------------------------------
        # EXISTING exclusion patterns (kept for coding/learning context)
        # -----------------------------------------------------------------------
        self.exclusion_patterns = [
            r'\b(coding help|conceptual doubt|course material)\b',
            r'\b(can someone help|how to|what is|how do|how does|help.*understand|help.*learn)\b.*\b(code|function|variable|class|python|java|javascript|program|algorithm|method|syntax|loops?|arrays?|string|object|recursion|data structure|sql|query)\b',
            r'\b(write|create|make|build).*\b(function|program|code|script|algorithm)\b.*\b(python|java|javascript|in)\b',
            r'\b(thank|thanks|great|awesome|helpful)\b',
            r'\b(when will|when is|when are|when does)\b.*\b(dsa|batch|session|class|module|start|begin|next)\b',
            r'\b(how many (modules|sessions|classes|questions))\b',
            r'\b(is there (a )?session|is there (a )?recording|is there (a )?repeat)\b',
            r'\b(which batch|what batch|my batch|our batch)\b',
            r'\b(what happens if|what if i|what should i do if)\b.*\b(can\'?t|fail|miss|attempt)\b',
            r'\b(sql|excel|tableau|sheet|assignment|query|data)\b.*\b(question|q\d|problem|help)\b',
        ]

    # -------------------------------------------------------------------------
    # Core detection helpers
    # -------------------------------------------------------------------------

    def _matches_hard_exclusion(self, message: str) -> bool:
        """
        Returns True if the message should be NEUTRAL regardless of negative keywords.
        Based on analysis of wrongly flagged messages (Document 3).
        """
        msg_lower = message.lower().strip()

        # Emoji presence
        if self.positive_emoji_pattern.search(message):
            return True

        # Onboarding / navigation confusion
        for pat in self.onboarding_confusion_patterns:
            if re.search(pat, msg_lower, re.IGNORECASE):
                return True

        # Pure schedule inquiry
        for pat in self.schedule_inquiry_patterns:
            if re.search(pat, msg_lower, re.IGNORECASE):
                return True

        # Instructor announcement
        for pat in self.instructor_announcement_patterns:
            if re.search(pat, msg_lower, re.IGNORECASE):
                return True

        # Support redirect template (needs 2+ signals to confirm template vs genuine complaint)
        support_redirect_hits = sum(
            1 for pat in self.support_redirect_templates
            if re.search(pat, msg_lower, re.IGNORECASE)
        )
        if support_redirect_hits >= 2:
            return True

        # Collaborative peer helping (only exclude if no very strong negative present)
        for pat in self.collaborative_patterns:
            if re.search(pat, msg_lower, re.IGNORECASE):
                if not any(re.search(p, msg_lower, re.IGNORECASE) for p in self.very_strong_negative):
                    return True

        # Positive/neutral phrases (only if no very strong negative present)
        for pat in self.positive_neutral_phrases:
            if re.search(pat, msg_lower, re.IGNORECASE):
                if not any(re.search(p, msg_lower, re.IGNORECASE) for p in self.very_strong_negative):
                    return True

        # Coding learning question (only if no very strong negative present)
        for pat in self.coding_learning_patterns:
            if re.search(pat, msg_lower, re.IGNORECASE):
                if not any(re.search(p, msg_lower, re.IGNORECASE) for p in self.very_strong_negative):
                    return True

        return False

    def _count_very_strong_signals(self, message_lower: str) -> int:
        return sum(
            1 for pat in self.very_strong_negative
            if re.search(pat, message_lower, re.IGNORECASE)
        )

    def _count_strong_signals(self, message_lower: str) -> int:
        return sum(
            1 for pat in self.strong_negative
            if re.search(pat, message_lower, re.IGNORECASE)
        )

    def _count_moderate_signals(self, message_lower: str) -> int:
        return sum(
            1 for pat in self.moderate_negative
            if re.search(pat, message_lower, re.IGNORECASE)
        )

    # -------------------------------------------------------------------------
    # Main analyze method
    # -------------------------------------------------------------------------

    def analyze(self, message_body: str) -> str:
        """
        Analyze message sentiment.

        Decision logic (priority order):
        1. Empty → neutral
        2. Hard exclusion match → neutral  (false positive prevention)
        3. Existing exclusion patterns → neutral  (coding/learning contexts)
        4. Very strong signal (1+) → negative
        5. Strong signal (1+) → negative
        6. Moderate signals (2+) → negative
        7. Context score (>= 3, raised threshold) → negative
        8. Default → neutral

        Args:
            message_body: The message text to analyze

        Returns:
            'negative' or 'neutral'
        """
        if not message_body or not message_body.strip():
            return 'neutral'

        # Step 1: Hard exclusion check
        if self._matches_hard_exclusion(message_body):
            return 'neutral'

        message_lower = message_body.lower()

        # Step 2: Existing exclusion patterns
        exclusion_match = any(
            re.search(pattern, message_lower, re.IGNORECASE)
            for pattern in self.exclusion_patterns
        )
        if exclusion_match:
            very_strong = self._count_very_strong_signals(message_lower)
            if very_strong == 0:
                return 'neutral'

        # Step 3: Signal counting
        very_strong = self._count_very_strong_signals(message_lower)
        strong = self._count_strong_signals(message_lower)
        moderate = self._count_moderate_signals(message_lower)

        # Step 4: Very strong signal alone is enough
        if very_strong >= 1:
            return 'negative'

        # Step 5: Strong signal alone is enough
        if strong >= 1:
            return 'negative'

        # Step 6: Moderate signals need at least 2
        if moderate >= 2:
            return 'negative'

        # Step 7: Context analysis fallback (threshold raised to 3)
        context_score = self._analyze_context(message_body)
        if context_score >= 3:
            return 'negative'

        return 'neutral'

    # -------------------------------------------------------------------------
    # Context analysis
    # -------------------------------------------------------------------------

    def _analyze_context(self, message_body: str) -> int:
        """
        Analyze message context for negative sentiment beyond pattern matching.

        Returns:
            Context score (3+ = negative at raised threshold)
        """
        if not message_body or not message_body.strip():
            return 0

        message_lower = message_body.lower()
        score = 0

        problem_words = [
            'problem', 'issue', 'error', 'fail', 'broken', 'wrong', 'bad',
            'dikkat', 'pareshani', 'mushkil', 'galat', 'kharab'
        ]
        emotion_words = [
            'frustrated', 'angry', 'disappointed', 'upset', 'sad', 'worried',
            'stressed', 'pareshan', 'gussa', 'tension', 'chinta'
            # NOTE: 'confused' removed — too many FPs from new-user confusion
        ]
        help_words = [
            'help', 'urgent', 'asap', 'immediately', 'priority', 'madad', 'jaldi'
            # NOTE: 'please' removed — too generic
        ]
        intensifiers = [
            'very', 'extremely', 'really', 'too', 'so', 'completely', 'totally',
            'bahut', 'bohot', 'kaafi', 'bilkul', 'poora'
        ]
        negation_words = [
            'not', 'no', 'never', 'none', 'nothing', 'nowhere', 'nobody',
            'nahi', 'nhi', 'nahin'
        ]
        positive_context = [
            'good', 'great', 'excellent', 'working', 'solved', 'fixed', 'thanks',
            'achha', 'badhiya', 'sahi', 'theek', 'thank', 'dhanyavaad'
        ]

        # Guard: short questions capped at 1 (never reach threshold of 3 alone)
        is_short_question = (
            len(message_lower.split()) <= 20
            and message_lower.strip().endswith('?')
        )

        words = message_lower.split()

        # 1. Problem + emotion combination (within 10 words)
        for i, word in enumerate(words):
            for problem in problem_words:
                if problem in word:
                    window_start = max(0, i - 10)
                    window_end = min(len(words), i + 10)
                    window = words[window_start:window_end]
                    for emotion in emotion_words:
                        if any(emotion in w for w in window):
                            score += 1
                            break

        # 2. Intensifier + problem/emotion combination
        for i in range(len(words) - 1):
            if any(intensifier in words[i] for intensifier in intensifiers):
                next_words = words[i + 1:min(i + 3, len(words))]
                if any(any(neg in w for neg in problem_words + emotion_words) for w in next_words):
                    score += 1

        # 3. Negation + positive context (e.g., "not good", "no help")
        for i in range(len(words) - 1):
            if any(negation in words[i] for negation in negation_words):
                next_words = words[i + 1:min(i + 4, len(words))]
                if any(any(pos in w for w in next_words) for pos in positive_context):
                    score += 2
                    break

        # 4. Problem + help-seeking combination
        has_problem = any(problem in message_lower for problem in problem_words)
        has_help = any(help_word in message_lower for help_word in help_words)
        if has_problem and has_help:
            score += 1

        # 5. Repeated negative themes
        for neg_word in problem_words + emotion_words:
            if message_lower.count(neg_word) >= 2:
                score += 1
                break

        # 6. Multiple question marks (raised from 2 to 3 to reduce FPs)
        if message_lower.count('?') >= 3:
            score += 1

        # 7. Urgent/escalation language with problems
        urgent_patterns = [
            r'\b(urgent|asap|immediately|priority)\b.*\b(help|issue|problem)',
            r'\b(jaldi|turant|abhi|urgent).*\b(help|madad|dikkat|problem)',
            r'\b(still|yet|already).*\b(not|no|nahi).*\b(working|fixed|resolved)',
            r'\b(waiting|waited|wait.*for).*\b(days|weeks|long|time)',
        ]
        for pattern in urgent_patterns:
            if re.search(pattern, message_lower):
                score += 1

        # 8. Lack of response/communication
        communication_patterns = [
            r'\b(no.*response|no.*reply|not.*responding|haven\'?t.*heard)',
            r'\b(nahi.*mila|nahi.*aaya|koi.*nahi).*\b(response|reply|jawab)',
            r'\b(tried|trying).*\b(reach|contact|call).*\b(but|no|not)',
        ]
        for pattern in communication_patterns:
            if re.search(pattern, message_lower):
                score += 1

        # 9. Time-related frustration
        time_frustration = [
            r'\b(still|yet|already).*\b(waiting|pending|not)',
            r'\b(how.*long|when.*will|why.*taking).*\b(time|long)',
            r'\b(days|weeks|months).*\b(no|not|nahi).*\b(response|update|reply)',
            r'\b(kab.*tak|kitne.*din|kitna.*time).*\b(lagega|wait)',
        ]
        for pattern in time_frustration:
            if re.search(pattern, message_lower):
                score += 1

        # 10. Consequence/impact statements (high weight)
        consequence_patterns = [
            r'\b(affecting|impacting|hurting|damaging).*\b(career|future|growth|progress)',
            r'\b(cannot|can\'?t).*\b(continue|proceed|move forward|cope)',
            r'\b(waste|wasting).*\b(time|money|effort|paise)',
            r'\b(regret|regretting|mistake).*\b(joining|enrolled|decision)',
        ]
        for pattern in consequence_patterns:
            if re.search(pattern, message_lower):
                score += 2

        if is_short_question:
            score = min(score, 1)

        return score

    # -------------------------------------------------------------------------
    # Debugging utility
    # -------------------------------------------------------------------------

    def get_matched_patterns(self, message_body: str) -> list:
        """Get list of matched signal categories for debugging."""
        if not message_body or not message_body.strip():
            return []

        message_lower = message_body.lower()
        matched = []

        if self._matches_hard_exclusion(message_body):
            matched.append('HARD_EXCLUSION')

        vs = self._count_very_strong_signals(message_lower)
        if vs > 0:
            matched.append(f'VERY_STRONG_SIGNALS({vs})')

        st = self._count_strong_signals(message_lower)
        if st > 0:
            matched.append(f'STRONG_SIGNALS({st})')

        mo = self._count_moderate_signals(message_lower)
        if mo > 0:
            matched.append(f'MODERATE_SIGNALS({mo})')

        ctx = self._analyze_context(message_body)
        if ctx > 0:
            matched.append(f'CONTEXT_SCORE({ctx})')

        return matched

    def _load_sentiment_context(self, filepath: str) -> dict:
        context = {
            'loaded': False,
            'critical_signals': [],
            'support_failures': [],
            'technical_issues': [],
            'negative_language': [],
            'exclusions': []
        }
        if not os.path.exists(filepath):
            return context
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                context['loaded'] = True
                context['raw_content'] = content
                if 'CRITICAL SIGNALS:' in content:
                    context['has_critical_signals'] = True
                if 'SUPPORT FAILURES:' in content:
                    context['has_support_failures'] = True
                if 'TECHNICAL ISSUES' in content:
                    context['has_technical_issues'] = True
                if 'NEGATIVE LANGUAGE' in content:
                    context['has_negative_language'] = True
        except Exception as e:
            print(f"Warning: Could not load sentiment.md: {e}")
        return context

    def _check_manual_examples(self, filepath: str) -> bool:
        return os.path.exists(filepath)

    def get_sentiment_rules_info(self) -> str:
        info_parts = []
        if self.sentiment_context.get('loaded'):
            info_parts.append("Sentiment rules loaded from sentiment.md")
            info_parts.append(f"- Critical Signals: {'✓' if self.sentiment_context.get('has_critical_signals') else '✗'}")
            info_parts.append(f"- Support Failures: {'✓' if self.sentiment_context.get('has_support_failures') else '✗'}")
            info_parts.append(f"- Technical Issues: {'✓' if self.sentiment_context.get('has_technical_issues') else '✗'}")
            info_parts.append(f"- Negative Language: {'✓' if self.sentiment_context.get('has_negative_language') else '✗'}")
        else:
            info_parts.append("Using built-in sentiment rules (sentiment.md not found)")
        if self.manual_examples_loaded:
            info_parts.append("✓ Manual negative examples loaded (sentimentManual.md)")
        return "\n".join(info_parts)
