ip = "192.168.99.121"
ip1 = "192.168.99.122"


class E:

    def __init__(self, service_name="app_mn1"):
        self.url_list = url_list = ["http://" + ip + ":666/~/mn-cse/mn-name/AE1/RFID_Container_for_stage4",
                 "http://" + ip1 + "/~/mn-cse/mn-name/AE2/Control_Command_Container",
                 "http://" + ip + ":1111/test", "http://" + ip1 + ":2222/test"]

e = E()
print(e.url_list)