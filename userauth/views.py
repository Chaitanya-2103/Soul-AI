import json
import jwt
from django.http import JsonResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .models import Profile
import datetime 


def generate_jwt(user):
    payload = {
        'user_id': user.id,
        'email': user.email,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=3600)
    }
    token = jwt.encode(payload, 'soulaijwt', algorithm='HS256')
    return token

@csrf_exempt
def signup(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        name = data.get('name')
        dob = data.get('dob')
        email = data.get('email')
        password = data.get('password')

        if User.objects.filter(username=email).exists():
            return JsonResponse({'error': 'User already exists'}, status=400)

        user = User.objects.create_user(username=email, email=email, password=password)
        Profile.objects.create(user=user, name=name, dob=dob)

        token = generate_jwt(user)
        return JsonResponse({'token': token}, status=201)

@csrf_exempt
def login_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        email = data.get('email')
        password = data.get('password')
        user = authenticate(username=email, password=password)

        if user:
            token = generate_jwt(user)
            return JsonResponse({'token': token}, status=200)
        else:
            return JsonResponse({'error': 'Invalid credentials'}, status=401)
