from channels.generic.websocket import AsyncWebsocketConsumer
import json
from channels.db import database_sync_to_async
from .models import EMF, DataItem

class DashboardConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        dashboard_slug = self.scope['url_route']['kwargs']['dashboard_slug']
        print(dashboard_slug)
        self.dashboard_slug = dashboard_slug
        self.room_group_name = f'EMF_dashboard-{dashboard_slug}'
        await self.channel_layer.group_add(
            self.room_group_name, self.channel_name
        )
        await self.accept()


    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        print(f"JSON: {text_data_json}")
        message = text_data_json['message']
        # sender = text_data_json['sender']
        # print(f'[{sender}] {message}')
        # print(text_data_json)

        await self.save_data_item(self.scope['user'], message, self.dashboard_slug)

        await self.channel_layer.group_send(
            self.room_group_name, {
                'type': 'dashboard_message',
                'message': message,
                # 'sender': sender
            }
        )

    async def dashboard_message(self, event):
        message = event['message']
        # sender = event['sender']
    
        await self.send(text_data=json.dumps({
            'message': message,
            # 'sender': sender
        }))

    async def disconnect(self, close_code):
        print(f'connection closed: {close_code}')

        await self.channel_layer.group_discard(
            self.room_group_name, self.channel_name
        )

    @database_sync_to_async
    def create_data_item(self, user, message, slug): # should be sender instead of user
        obj = EMF.objects.get(slug=slug)
        return DataItem.objects.create(
            emf=obj, value=message, owner=user
        )

    async def save_data_item(self, user, message, slug):
        await self.create_data_item(user, message, slug)