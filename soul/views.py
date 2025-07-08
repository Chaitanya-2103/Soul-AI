from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
import json
#from .utils import fetch_wikipedia_info,analyze_text
from .utils import smart_summary_with_open_fallback
from django.views.decorators.csrf import csrf_exempt

# Create your views here.

def hello(request):
    if ( request.method=='GET' ):
        return JsonResponse({'data':"'Hello , i am your Soul'"})


@csrf_exempt
def info(request):
    if request.method == "POST": 
        try:
            data = json.loads(request.body)  
            text = data.get("query", "") 
            response=smart_summary_with_open_fallback(text)
            return JsonResponse(response)
        except json.JSONDecodeError: 
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    return JsonResponse({'error': 'Only POST requests allowed'}, status=405)  