from django.http import HttpResponse
from.otp_verify import verify_otp


def verify_code_view(request):
    if request.method == "POST":
        otp_input = request.POST.get("otp_code")

        if verify_otp(request.user, otp_input):
            return HttpResponse("✅ Access Granted")
        return HttpResponse("❌ Invalid or Expired OTP", status=403)