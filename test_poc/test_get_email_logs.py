from __future__ import print_function
import time
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
from pprint import pprint

configuration = sib_api_v3_sdk.Configuration()
configuration.api_key['api-key'] = ''

api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))
limit = 2500
offset = 0
start_date = '2024-01-03'
end_date = '2024-01-04'

try:
    api_response = api_instance.get_email_event_report(limit=limit, offset=offset, start_date=start_date, end_date=end_date, sort='desc' )
    pprint(api_response)
except ApiException as e:
    print("Exception when calling SMTPApi->get_email_event_report: %s\n" % e)