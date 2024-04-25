# -*- coding: utf-8 -*-
"""
@author: tantrieu31
"""


from hubspot import HubSpot
from hubspot.crm.contacts import ApiException

from hubspot.crm.contacts import SimplePublicObjectInputForCreate, PublicObjectSearchRequest
from hubspot.crm.contacts.exceptions import ApiException

from dotenv import load_dotenv
import os
import json


def load_hubspot_access_token():
    load_dotenv()
    return os.environ['HUBSPOT_ACCESS_TOKEN']

# load API client
hubspot_api_client = HubSpot(access_token=load_hubspot_access_token())


def upsert_contact(contact_data):
    email = contact_data['email']
    # 1. Try to find an existing contact:
    public_object_search_request = PublicObjectSearchRequest(
        filter_groups=[
            {
                "filters": [
                    {
                        "value": email,
                        "propertyName": "email",
                        "operator": "EQ"
                    }
                ]
            }
        ], limit=1
    )
    search_results = hubspot_api_client.crm.contacts.search_api.do_search(
        public_object_search_request=public_object_search_request)

    record = SimplePublicObjectInputForCreate(
        properties=contact_data
    )
    if search_results.total > 0:  # Contact found
        contact_id = search_results.results[0].id
        res = hubspot_api_client.crm.contacts.basic_api.update(
            contact_id, simple_public_object_input=record)

    else:  # Contact not found
        res = hubspot_api_client.crm.contacts.basic_api.create(
            simple_public_object_input=record)
    return int(res.id)


def save_hubspot_contact(contact_data):
    try:
        # get and delete hubspot_contact_id
        hsp_contact_id = contact_data['hubspot_contact_id']
        del contact_data['hubspot_contact_id']

        record = SimplePublicObjectInputForCreate(
            properties=contact_data
        )

        if hsp_contact_id > 0:
            res = hubspot_api_client.crm.contacts.basic_api.update(
                simple_public_object_input=record, contact_id=hsp_contact_id)
            return int(res.id)
        else:
            return upsert_contact(contact_data)

    except ApiException as e:
        # print("Exception when creating contact: %s\n" % e)
        error_message = json.loads(e.body)
        print("Exception when creating contact: %s\n" %
              error_message['message'])
        if e.status == 409:
            return 0
    return -1


# Sample Contact Data (replace with your actual data)
contact_data = {
    'email': 'tantrieuf31.database@gmail.com',
    'firstname': 'Trieu',
    'lastname': 'Nguyen',
    'phone': '555-555-5555',
    'jobtitle': 'Tester',
    'lifecyclestage': 'lead',
    'cdp_data_labels': 'demo',
    'hubspot_contact_id': 0
}

# set a valid hubspot_contact_id to test update
contact_data['hubspot_contact_id'] = 0

# test save API
contact_id = save_hubspot_contact(contact_data)

if contact_id > 0:
    print("Contact created successfully. Contact ID: " + str(contact_id))
elif contact_id == 0:
    print("Contact already exists." + str(contact_id))
else:
    print("Contact creation failed." + str(contact_id))
