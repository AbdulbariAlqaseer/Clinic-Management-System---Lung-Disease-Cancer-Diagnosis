from django.utils.deprecation import MiddlewareMixin

class DisableCSRFMiddleware(MiddlewareMixin):
    def process_view(self, request, view_func, view_args, view_kwargs):
        # if request.path.startswith('api/accounts/auth/login/'):  # Adjust the path prefix to match your API URL pattern
        setattr(request, '_dont_enforce_csrf_checks', True)