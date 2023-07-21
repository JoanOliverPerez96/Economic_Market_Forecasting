from django.shortcuts import render, redirect, get_object_or_404
from .models import EMF, DataItem, MLData
from faker import Faker
from django.http import JsonResponse
from django.db.models import Sum

fake = Faker()

def main(request):
    qs = EMF.objects.all()
    if request.method == 'POST':
        new_dash = request.POST.get('new-dashboard')
        obj, _ = EMF.objects.get_or_create(name=new_dash)
        return redirect("EMF_dashboard:dashboard", slug=obj.slug)
    return render(request, 'EMF_dashboard/main.html',{'qs':qs})

def dashboard(request, slug):
    obj = get_object_or_404(EMF, slug=slug)
    return render(request, 'EMF_dashboard/dashboard.html',{
        'name': obj.name,
        'slug': obj.slug,
        'data': obj.data,
        # 'datepicker': obj.datepicker,
        'user': request.user.username if request.user.username else fake.name()
    })

def chart_data(request, slug):
    obj = get_object_or_404(EMF, slug=slug)
    qs = obj.data.values('owner').annotate(Sum('value'))
    chart_data = [x["value__sum"] for x in qs]
    chart_labels = [x["owner"] for x in qs]
    return JsonResponse({
        "chartData": chart_data,
        "chartLabels": chart_labels
    })

# def datepicker(request, slug, dataitem):
#     obj = get_object_or_404(EMF, slug=slug, dataitem=dataitem)
#     return render(request, 'EMF_dashboard/chart_data.html',{
#         'datepicker': obj.datepicker,
#     })


    # return render(request, 'EMF_dashboard/chart_data.html',{
    #     'name': obj.name,
    #     'slug': obj.slug,
    #     'data': obj.data,
    #     'user': request.user.username if request.user.username else fake.name()
    # })