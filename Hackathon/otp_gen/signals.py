from django.db.models.signals import post_save
from django.dispatch import receiver
from.models import IncomingData
from.otp_service import generate_and_send_otp


@receiver(post_save, sender=IncomingData)
def send_otp_on_db_trigger(sender, instance, created, **kwargs):
    if created:
        generate_and_send_otp(instance.user)