from django.shortcuts import render

# Create your views here.

def predict_price(request):
    return render(request, "prediction.html")
