from mtgsdk import Card

cards = Card.where(name="Felidar Retreat").all()

for card in cards:
    print(card.name + ': set ' + card.set)
