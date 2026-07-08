from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from authoringtool.models import Simulation, ExperimentLL, VRARExperiment
from .admin_views import staff_required


def _qr_url(vr):
    try:
        return vr.qr_code.url if vr.qr_code else None
    except Exception:
        return None


# ── Simulations ────────────────────────────────────────────────────────

@require_POST
@staff_required
def admin_create_simulation(request):
    name = request.POST.get('name', '').strip()
    iframe_url = request.POST.get('iframe_url', '').strip()
    if not name or not iframe_url:
        return JsonResponse({'success': False, 'error': 'Name and iframe URL are required.'})
    try:
        width = int(request.POST.get('width') or 800)
        height = int(request.POST.get('height') or 600)
    except ValueError:
        return JsonResponse({'success': False, 'error': 'Width and height must be numbers.'})
    allow_fullscreen = request.POST.get('allow_fullscreen') == 'true'
    sim = Simulation.objects.create(
        name=name, iframe_url=iframe_url,
        width=width, height=height, allow_fullscreen=allow_fullscreen,
    )
    return JsonResponse({
        'success': True, 'id': sim.id, 'name': sim.name,
        'iframe_url': sim.iframe_url, 'width': sim.width,
        'height': sim.height, 'allow_fullscreen': sim.allow_fullscreen,
    })


@require_POST
@staff_required
def admin_edit_simulation(request, sim_id):
    sim = get_object_or_404(Simulation, id=sim_id)
    name = request.POST.get('name', '').strip()
    iframe_url = request.POST.get('iframe_url', '').strip()
    if not name or not iframe_url:
        return JsonResponse({'success': False, 'error': 'Name and iframe URL are required.'})
    try:
        width = int(request.POST.get('width') or 800)
        height = int(request.POST.get('height') or 600)
    except ValueError:
        return JsonResponse({'success': False, 'error': 'Width and height must be numbers.'})
    sim.name = name
    sim.iframe_url = iframe_url
    sim.width = width
    sim.height = height
    sim.allow_fullscreen = request.POST.get('allow_fullscreen') == 'true'
    sim.save()
    return JsonResponse({
        'success': True, 'id': sim.id, 'name': sim.name,
        'iframe_url': sim.iframe_url, 'width': sim.width,
        'height': sim.height, 'allow_fullscreen': sim.allow_fullscreen,
    })


@require_POST
@staff_required
def admin_delete_simulation(request, sim_id):
    sim = get_object_or_404(Simulation, id=sim_id)
    sim.delete()
    return JsonResponse({'success': True})


# ── Remote Labs (LabsLand) ─────────────────────────────────────────────

@require_POST
@staff_required
def admin_create_remote_lab(request):
    name = request.POST.get('name', '').strip()
    launch_url = request.POST.get('launch_url', '').strip()
    consumer_key = request.POST.get('consumer_key', '').strip()
    shared_secret = request.POST.get('shared_secret', '').strip()
    if not name or not launch_url or not consumer_key or not shared_secret:
        return JsonResponse({'success': False, 'error': 'Name, launch URL, consumer key, and shared secret are required.'})
    lab = ExperimentLL.objects.create(
        name=name,
        description=request.POST.get('description', '').strip(),
        launch_url=launch_url,
        consumer_key=consumer_key,
        shared_secret=shared_secret,
    )
    return JsonResponse({
        'success': True, 'id': lab.id, 'name': lab.name,
        'description': lab.description, 'launch_url': lab.launch_url,
        'consumer_key': lab.consumer_key,
        # shared_secret intentionally omitted
    })


@require_POST
@staff_required
def admin_edit_remote_lab(request, lab_id):
    lab = get_object_or_404(ExperimentLL, id=lab_id)
    name = request.POST.get('name', '').strip()
    launch_url = request.POST.get('launch_url', '').strip()
    consumer_key = request.POST.get('consumer_key', '').strip()
    if not name or not launch_url or not consumer_key:
        return JsonResponse({'success': False, 'error': 'Name, launch URL, and consumer key are required.'})
    shared_secret = request.POST.get('shared_secret', '').strip()
    lab.name = name
    lab.description = request.POST.get('description', '').strip()
    lab.launch_url = launch_url
    lab.consumer_key = consumer_key
    if shared_secret:          # blank = keep existing
        lab.shared_secret = shared_secret
    lab.save()
    return JsonResponse({
        'success': True, 'id': lab.id, 'name': lab.name,
        'description': lab.description, 'launch_url': lab.launch_url,
        'consumer_key': lab.consumer_key,
        # shared_secret intentionally omitted
    })


@require_POST
@staff_required
def admin_delete_remote_lab(request, lab_id):
    lab = get_object_or_404(ExperimentLL, id=lab_id)
    lab.delete()
    return JsonResponse({'success': True})


# ── VR/AR Labs ────────────────────────────────────────────────────────

@require_POST
@staff_required
def admin_create_vr_lab(request):
    name = request.POST.get('name', '').strip()
    launch_url = request.POST.get('launch_url', '').strip()
    if not name or not launch_url:
        return JsonResponse({'success': False, 'error': 'Name and launch URL are required.'})
    vr = VRARExperiment(
        name=name,
        description=request.POST.get('description', '').strip(),
        launch_url=launch_url,
    )
    vr.save()   # triggers QR generation
    return JsonResponse({
        'success': True, 'id': vr.id, 'name': vr.name,
        'description': vr.description, 'launch_url': vr.launch_url,
        'qr_code_url': _qr_url(vr),
    })


@require_POST
@staff_required
def admin_edit_vr_lab(request, vr_id):
    vr = get_object_or_404(VRARExperiment, id=vr_id)
    name = request.POST.get('name', '').strip()
    launch_url = request.POST.get('launch_url', '').strip()
    if not name or not launch_url:
        return JsonResponse({'success': False, 'error': 'Name and launch URL are required.'})
    # If URL changed, clear old QR so save() regenerates it
    if launch_url != vr.launch_url and vr.qr_code:
        vr.qr_code.delete(save=False)
        vr.qr_code = None
    vr.name = name
    vr.description = request.POST.get('description', '').strip()
    vr.launch_url = launch_url
    vr.save()
    return JsonResponse({
        'success': True, 'id': vr.id, 'name': vr.name,
        'description': vr.description, 'launch_url': vr.launch_url,
        'qr_code_url': _qr_url(vr),
    })


@require_POST
@staff_required
def admin_delete_vr_lab(request, vr_id):
    vr = get_object_or_404(VRARExperiment, id=vr_id)
    vr.delete()
    return JsonResponse({'success': True})
