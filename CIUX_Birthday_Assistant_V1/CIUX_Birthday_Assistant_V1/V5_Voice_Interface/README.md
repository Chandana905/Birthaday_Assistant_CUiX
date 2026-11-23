# CIUX Birthday Assistant — Version 5

**Voice Conversational Interface**  
GM University · UE24CS3540 — Conversational Interfaces and UX

V5 is a separate copy of V4 that adds browser-based voice input and spoken
responses. It keeps V4's chat API and conversational engine: NLU, optional
Groq parsing, rule-based fallback, workbook grounding, dialogue state,
clarification, and reset. V5 does not upload or process raw audio on the Flask
server.

## V4 versus V5

- **V4** accepts typed messages and returns grounded text answers.
- **V5** retains the V4 behavior and adds browser Automatic Speech Recognition
  (ASR) and optional Text-to-Speech (TTS).
- Recognized speech is placed in the normal question field and sent through
  the same `/api/chat` endpoint as typed messages.
- V5 uses a distinct Flask session cookie so its dialogue state does not
  collide with another version running on the same host.
- All birthday and location facts continue to come from `data/people.xlsx`.

## Voice interaction

### Speech recognition (ASR)

Click **Speak**, allow microphone access, and speak the complete question.
V5 uses the browser's built-in `SpeechRecognition` or
`webkitSpeechRecognition` API with `en-IN`, continuous recognition, and
interim results. The input field previews recognized words as you speak while
final transcript segments are accumulated. Nothing is sent to the backend
until you click **Stop Speaking** and recognition ends; then the complete
transcript is submitted once through the existing `sendMsg()` chat function.
Press **Escape** to cancel without sending the recording. If recognition is
interrupted, its final transcript is retained in the input but is not sent.

Speech recognition is not perfect. Indian names, place names, relationship
terms, month names, accents, background noise, microphone quality, and browser
speech services can cause transcription errors. Check the transcribed text in
the conversation; if it is wrong, type a correction. A transcription mistake
is different from an NLU or data-grounding mistake.

### Spoken answers (TTS)

Select **Read responses aloud** to hear chatbot answers. The browser's
`speechSynthesis` and `SpeechSynthesisUtterance` APIs speak only the answer
text, not the debug panel. Text answers remain visible. A natural available
English voice is preferred, with an Indian English voice preferred when
available. Unchecking the option cancels any current speech.

## Browser and microphone requirements

Voice features depend on browser support and installed speech services.
Chromium-based browsers generally provide Web Speech APIs, but availability
varies by browser, operating system, and locale. If speech recognition is
unsupported, typed chat continues to work and V5 displays a notice. If speech
synthesis is unavailable, the read-aloud checkbox is disabled and answers
remain visible as text.

Browsers generally allow microphone access only from a secure context:
**localhost** or an **HTTPS** site. Permission-denied, missing microphone,
no-speech, network, and recognition errors are surfaced in the interface.
In particular, Chrome's `network` recognition error means its online speech
service could not be reached; it is not an error from V5's Flask chat endpoint.
This service may be unavailable because of connectivity, VPN/firewall rules,
or browser/organization policy, and the application cannot restore that
external service. Text chat remains available in all cases.

## Install and run (Python 3.10)

Open the `V5_Voice_Interface` folder in VS Code, then use its integrated
PowerShell terminal:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit the V5 root `.env` and configure Groq only if you want LLM-assisted NLU:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
FLASK_SECRET=replace_with_a_private_random_secret
```

Never put credentials in source code or browser JavaScript. `.env` is ignored
by Git. If no Groq key is configured or the service fails, V5's rule-based NLU
fallback remains available.
If `FLASK_SECRET` is omitted, V5 warns and uses a temporary random development
secret; configure it in `.env` to keep session cookies valid across restarts.

Person_ID questions such as `Show birthday of P001`, `Where does P001 stay`,
`Tell me about P001`, and `Who is P001` are resolved only against the Excel
workbook. Entering only `P001` asks which workbook details you want.

Start the application:

```powershell
python -c "from app import app; app.run(host='127.0.0.1', port=5001)"
```

Open <http://127.0.0.1:5001>. This port avoids conflict with an existing V4
server on port 5000. Use the microphone button on localhost and allow the
browser's microphone permission prompt. If port 5000 is free, `python app.py`
also starts V5 there.

## Try it

Try typing or speaking these questions:

1. `When is Rahul's birthday?`
2. `Do I need to wish any nephew during September?`
3. `What about October?` (after the previous question)
4. `Where does Rahul stay?`
5. `How many birthdays are there next month?`
6. `When is my nephew's birthday?`

The assistant may ask for a name when a relationship query matches multiple
people. Reply with a name from the clarification prompt. Check the recognized
transcript to distinguish ASR errors from NLU or grounding errors.

## Tests

Run the inherited backend tests from this folder:

```powershell
py -3.10 -m pytest -q
```

The tests do not require a microphone or live Groq credentials. They exercise
V4's grounding, fallback, dialogue state, clarification, reset, and API routes.
Browser speech APIs are feature-detected at runtime and require a compatible
browser for manual voice testing.

## Security and limitations

- Voice audio is handled by the browser's built-in recognition; it is not
  posted to a new Flask voice endpoint.
- Optional Groq API credentials are used only by the server and are never
  returned to the browser.
- The default Flask secret is for local development; set a private
  `FLASK_SECRET` for deployment.
- Speech quality and API availability vary. Voice input is an accessibility
  option, not a guarantee of perfect transcription.
