class OneToManyTable:

    def __init__(self, name):
        """
        Initializes a OneToManyTable instance.

        :param name: The name of the table.
        """
        self.name = name
        self.data = {}

    def add(self, item, related_items, strengths, position=None):
        """
        Accepts the item and one of its relationships.

        :param item: The main item to be added.
        :param related_items: A list of related items.
        :param strengths: A list of relationship strengths.
        :param position: Optional position information.
        """
        if item.name not in self.data:
            assert isinstance(related_items, list)
            assert isinstance(strengths, list)
            self.data[item.name] = {
                'obj': item,
                'list': related_items,
                'strengths': strengths,
                'position': [position],
                'count': [1]
            }
        else:
            raise Exception('Item already in table')

    def get(self, item):
        """
        Retrieves the data associated with the given item.

        :param item: The item to retrieve.
        :return: The data associated with the item.
        """
        return self.data[item.name]

    def remove(self, item):
        """
        Removes the given item from the table.

        :param item: The item to remove.
        """
        del self.data[item.name]

    def add_related_item(self, item, related_item, strength=1, position=None):
        """
        Adds a related item to the given item.

        :param item: The main item.
        :param related_item: The related item to add.
        :param strength: The strength of the relationship (default: 1).
        :param position: Optional position information.
        """
        if not self.data.get(item.name):
            self.add(item, [related_item], [1], position=position)
        else:
            if related_item not in self.data[item.name]['list']:
                self.data[item.name]['list'].append(related_item)
                self.data[item.name]['strengths'].append(strength)
                self.data[item.name]['position'].append(position)
                self.data[item.name]['count'].append(1)
            else:
                raise Exception('The item is already related.')

        self._normalize_item(item)

    def remove_related_item(self, item, related_item):
        """
        Removes a related item from the given item.

        :param item: The main item.
        :param related_item: The related item to remove.
        """
        idx = self.data[item.name]['list'].index(related_item)
        self.data[item.name]['list'].remove(related_item)
        self.data[item.name]['strengths'].pop(idx)
        self.data[item.name]['position'].pop(idx)
        self.data[item.name]['count'].pop(idx)

        # If the list now has 0 items in it, we should not normalize.
        # Because normalizing will intentionally throw an exception.
        if len(self.data[item.name]['list']) > 0:
            self._normalize_item(item)

    def verify_data_integrity(self):
        """
        Verifies the integrity of the data in the table.
        """
        for item_name, val in self.data.items():  # Use .items() instead of .iteritems()
            item_list = val['list']

            # Verify that the length of the node_list and strengths are the same
            assert len(item_list) == len(val['strengths'])

            if len(item_list) > 0:
                assert 0.99 < sum(val['strengths']) <= 1.01

    def _normalize_item(self, item):
        """
        Normalizes the strengths of the relationships for the given item.

        :param item: The item to normalize.
        """
        self.data[item.name]['strengths'][:] = self._normalize(self.data[item.name]['strengths'])

    @staticmethod
    def _normalize(list_to_n):
        """
        Normalizes a list of values in place.

        :param list_to_n: The list to normalize.
        :return: The normalized list.
        """
        total = sum(list_to_n)

        if total <= 0.0:
            raise Exception('Cannot normalize a list with a total of zero or less.')

        list_to_n[:] = [item / total for item in list_to_n]
        return list_to_n

    def is_related(self, item, related_item):
        """
        Checks if the given item is related to the related_item.

        :param item: The main item.
        :param related_item: The related item to check.
        :return: True if related, False otherwise.
        """
        if not self.data.get(item.name):
            return False
        else:
            return bool(related_item in self.data[item.name]['list'])

    def increase_relationship_strength(self, item, related_item, amount, position=None):
        """
        Increases the strength of the relationship between the given item and related_item.

        :param item: The main item.
        :param related_item: The related item.
        :param amount: The quantity to increase the strength by.
        :param position: Optional position information.
        """
        assert self.is_related(item, related_item)

        idx = self.data[item.name]['list'].index(related_item)
        self.data[item.name]['strengths'][idx] += amount
        self.data[item.name]['count'][idx] += 1

        # Calculate the moving average of position
        if self.data[item.name]['position'][idx] is not None:
            old_pos = self.data[item.name]['position'][idx]
            new_pos = old_pos + (position - old_pos) / self.data[item.name]['count'][idx]
            self.data[item.name]['position'][idx] = new_pos
        else:
            self.data[item.name]['position'][idx] = position

        self._normalize_item(item)

    def count(self):
        """
        Returns the number of items in the table.

        :return: The count of items.
        """
        return len(self.data)

    def get_items_without_related_items(self):
        """
        Gets all items that are not related to anything.

        :return: A list of items without related items.
        """
        return [data['obj'] for data in self.data.values() if len(data['list']) == 0]