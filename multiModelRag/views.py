from django.shortcuts import render
from django.http import JsonResponse
from .langchain_helper import process_pdf, generate_response
from uuid import uuid4
import os

# In-memory storage for retrievers per session
retrievers = {}

# -----------------------------
# Chat page (frontend)
# -----------------------------
def chat_page(request):
    return render(request, "chat.html")  # Your HTML frontend template

# -----------------------------
# API endpoint to upload PDF
# -----------------------------
def upload_pdf(request):
    if request.method == "POST" and request.FILES.get("pdf_file"):
        pdf_file = request.FILES["pdf_file"]
        temp_path = f"temp_{uuid4().hex}.pdf"

        # Save uploaded PDF temporarily
        with open(temp_path, "wb") as f:
            for chunk in pdf_file.chunks():
                f.write(chunk)

        # Process PDF to create FAISS retriever
        retriever = process_pdf(temp_path)

        # Remove temp file
        os.remove(temp_path)

        # Create a session ID
        session_id = str(uuid4())
        retrievers[session_id] = retriever

        return JsonResponse({"success": True, "session_id": session_id})
    return JsonResponse({"success": False, "error": "No PDF uploaded."}, status=400)

# -----------------------------
# API endpoint to query chatbot
# -----------------------------
def chat_api(request):
    if request.method == "POST":
        data = request.POST
        session_id = data.get("session_id")
        user_query = data.get("query")
        model_key = data.get("model_key", "Deepseek-R1-distill-llama-70b")

        if not session_id or not user_query:
            return JsonResponse({"success": False, "error": "Missing session_id or query"}, status=400)

        retriever = retrievers.get(session_id)
        if not retriever:
            return JsonResponse({"success": False, "error": "Invalid session_id"}, status=400)

        # Generate response from LangChain
        try:
            response = generate_response(model_key, retriever, user_query)
            return JsonResponse({"success": True, "response": response})
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)}, status=500)

    return JsonResponse({"success": False, "error": "Invalid request method"}, status=405)

# -----------------------------
# Optional: Chat history (per session)
# -----------------------------
chat_history = {}

def get_chat_history(request, session_id):
    history = chat_history.get(session_id, [])
    return JsonResponse({"success": True, "history": history})
