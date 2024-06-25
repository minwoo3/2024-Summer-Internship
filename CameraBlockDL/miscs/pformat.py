
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# 터미널 메세지의 색 변경
def pprint(msg, options=[]):
    if len(options) == 0:
        print(msg)
        return None

    for option in options:
        option = option.upper()
        # getattr(object,'name') object 내부의 name이라는 멤버를 호출
        msg = f"{getattr(bcolors, option)}{msg}"

    print(f"{msg}{bcolors.ENDC}")
    return None


