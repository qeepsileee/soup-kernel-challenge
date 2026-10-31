import gc
from types import SimpleNamespace
import unittest
from solution.resource_audit import SavedInventory


class FakeStorage:
    def __init__(self, ptr, size):
        self.ptr, self.size = ptr, size
    def data_ptr(self):
        return self.ptr
    def nbytes(self):
        return self.size


class FakeTensor:
    is_cuda = True
    dtype = "float16"
    device = SimpleNamespace(index=0)
    def __init__(self, storage, shape):
        self.storage, self.shape = storage, shape
    def untyped_storage(self):
        return self.storage
    def numel(self):
        return 16


class InventoryAccountingTests(unittest.TestCase):
    def test_shared_storage_is_counted_once_and_released_after_last_view(self):
        inventory = SavedInventory(None)
        storage = FakeStorage(123, 64)
        a, b = FakeTensor(storage, (4, 4)), FakeTensor(storage, (2, 4))
        saved_a, saved_b = inventory.pack(a), inventory.pack(b)
        self.assertEqual(inventory.live_bytes, 64)
        self.assertIs(inventory.unpack(saved_a), a)
        del saved_a
        gc.collect()
        self.assertEqual(inventory.live_bytes, 64)
        del saved_b
        gc.collect()
        self.assertEqual(inventory.live_bytes, 0)
        self.assertEqual(inventory.peak_bytes, 64)
        self.assertEqual(sum(row["bytes"] for row in inventory.groups()), 64)

    def test_parameter_alias_is_not_an_activation_allocation(self):
        inventory = SavedInventory(None)
        tensor = FakeTensor(FakeStorage(456, 128), (8, 8))
        inventory.parameters.add((0, 456))
        self.assertIs(inventory.pack(tensor), tensor)
        self.assertEqual(inventory.peak_bytes, 0)
