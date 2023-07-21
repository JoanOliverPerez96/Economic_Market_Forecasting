from django.shortcuts import render

def Home(request):
    return render(request, 'index.html')

def User(request):
    username = request.GET['username']
    return render(request, 'user.html', {'name': username})

def Target(request):
    target = request.GET['target']
    return render(request, 'dashboard.html', {'target': target})