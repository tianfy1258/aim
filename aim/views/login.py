from aim.models import User
from aim.utils import *
import time



def login(request):
    if request.method == 'POST':
        req = json.loads(request.body)

        username = req['username']
        password = req['password']
        user_auth = req['isManager']
        user_auth = 'manager' if user_auth else 'user'

        response = {}

        user = User.objects.filter(username=username, password=password, user_auth=user_auth).first()
        if user:

            # token - session 模块
            if not request.session.session_key:
                request.session.save()
                LOGGER.info("新会话")

            request.session.set_expiry(3600 * 2)
            token = md5((username + password + "1258" + str(time.time())))
            request.session["token"] = token
            request.session["user_id"] = user.user_id
            response['token'] = token
            request.session.save()

            # return the userinfo to the client
            response["user"] = {
                "user_id": user.user_id,
                "username": user.username,
                "user_auth": user.user_auth,
            }

            return success_response(response)

        else:
            return error_response(response, "用户名或密码错误")


def logout(request):
    request.session.clear()
    response = {}
    return success_response(response)


def valid_token(request):
    req = json.loads(request.body)
    req_token = req['token']
    token = request.session.get("token")

    res = {}
    LOGGER.debug(f"request.session.session_key:{request.session.session_key}")
    LOGGER.debug(f"req_token:{req_token}")
    LOGGER.debug(f"token:{token}")
    if not token or token != req_token:
        return invalid_response(res, "登录信息失效，请重新登录")
    else:
        return success_response(res)
