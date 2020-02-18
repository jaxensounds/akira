import discord
import json

with open('./config.json') as configfile:
    config = json.load(configfile)

class _Client(discord.Client()):
    async def on_ready(self):
        print('Logged in as {0}!'.format(self.user))

client = _Client()
client.run(config['token'])
