from django.db import models
from django.contrib.auth.models import User

COUNTRY_CHOICES = [
    ('', '— Select country —'),
    ('Afghanistan', 'Afghanistan'), ('Albania', 'Albania'), ('Algeria', 'Algeria'),
    ('Argentina', 'Argentina'), ('Armenia', 'Armenia'), ('Australia', 'Australia'),
    ('Austria', 'Austria'), ('Azerbaijan', 'Azerbaijan'), ('Bangladesh', 'Bangladesh'),
    ('Belarus', 'Belarus'), ('Belgium', 'Belgium'), ('Bolivia', 'Bolivia'),
    ('Bosnia and Herzegovina', 'Bosnia and Herzegovina'), ('Brazil', 'Brazil'),
    ('Bulgaria', 'Bulgaria'), ('Cambodia', 'Cambodia'), ('Canada', 'Canada'),
    ('Chile', 'Chile'), ('China', 'China'), ('Colombia', 'Colombia'),
    ('Croatia', 'Croatia'), ('Cyprus', 'Cyprus'), ('Czech Republic', 'Czech Republic'),
    ('Denmark', 'Denmark'), ('Ecuador', 'Ecuador'), ('Egypt', 'Egypt'),
    ('Estonia', 'Estonia'), ('Ethiopia', 'Ethiopia'), ('Finland', 'Finland'),
    ('France', 'France'), ('Georgia', 'Georgia'), ('Germany', 'Germany'),
    ('Ghana', 'Ghana'), ('Greece', 'Greece'), ('Guatemala', 'Guatemala'),
    ('Hungary', 'Hungary'), ('India', 'India'), ('Indonesia', 'Indonesia'),
    ('Iran', 'Iran'), ('Iraq', 'Iraq'), ('Ireland', 'Ireland'),
    ('Israel', 'Israel'), ('Italy', 'Italy'), ('Japan', 'Japan'),
    ('Jordan', 'Jordan'), ('Kazakhstan', 'Kazakhstan'), ('Kenya', 'Kenya'),
    ('Kosovo', 'Kosovo'), ('Latvia', 'Latvia'), ('Lebanon', 'Lebanon'),
    ('Lithuania', 'Lithuania'), ('Luxembourg', 'Luxembourg'), ('Malaysia', 'Malaysia'),
    ('Mexico', 'Mexico'), ('Moldova', 'Moldova'), ('Morocco', 'Morocco'),
    ('Netherlands', 'Netherlands'), ('New Zealand', 'New Zealand'), ('Nigeria', 'Nigeria'),
    ('North Macedonia', 'North Macedonia'), ('Norway', 'Norway'), ('Pakistan', 'Pakistan'),
    ('Palestine', 'Palestine'), ('Peru', 'Peru'), ('Philippines', 'Philippines'),
    ('Poland', 'Poland'), ('Portugal', 'Portugal'), ('Romania', 'Romania'),
    ('Russia', 'Russia'), ('Saudi Arabia', 'Saudi Arabia'), ('Serbia', 'Serbia'),
    ('Singapore', 'Singapore'), ('Slovakia', 'Slovakia'), ('Slovenia', 'Slovenia'),
    ('South Africa', 'South Africa'), ('South Korea', 'South Korea'), ('Spain', 'Spain'),
    ('Sri Lanka', 'Sri Lanka'), ('Sweden', 'Sweden'), ('Switzerland', 'Switzerland'),
    ('Syria', 'Syria'), ('Taiwan', 'Taiwan'), ('Thailand', 'Thailand'),
    ('Tunisia', 'Tunisia'), ('Turkey', 'Turkey'), ('Ukraine', 'Ukraine'),
    ('United Arab Emirates', 'United Arab Emirates'), ('United Kingdom', 'United Kingdom'),
    ('United States', 'United States'), ('Uruguay', 'Uruguay'), ('Uzbekistan', 'Uzbekistan'),
    ('Venezuela', 'Venezuela'), ('Vietnam', 'Vietnam'), ('Yemen', 'Yemen'),
    ('Zimbabwe', 'Zimbabwe'),
]


class UserProfile(models.Model):
    user        = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    country     = models.CharField(max_length=100, blank=True, choices=COUNTRY_CHOICES)
    institution = models.CharField(max_length=255, blank=True, help_text="School or university you work at")
    bio         = models.TextField(max_length=500, blank=True, help_text="A short bio (max 500 characters)")

    def __str__(self):
        return f"Profile of {self.user.username}"
