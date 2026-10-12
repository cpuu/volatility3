"""Synthetic translation/stacking regressions; these do not replace real dumps."""

import struct
from types import SimpleNamespace

import pytest

from volatility3.framework import contexts, exceptions
from volatility3.framework.automagic.linux import (
    LinuxIntelSubStacker,
    LinuxIntelVMCOREINFOStacker,
    LinuxStacker,
)
from volatility3.framework.layers import arm, intel, physical


BANNER = b"Linux version synthetic integration test\x00"


def memory_context(buffer):
    context = contexts.Context()
    context.layers.add_layer(
        physical.BufferDataLayer(context, "physical", "physical", bytes(buffer))
    )
    context.config["test.memory_layer"] = "physical"
    context.config["test.page_map_offset"] = 0x1000
    return context


def intel_memory(kind):
    """Hand-encoded two-page mappings with a deliberately noncontiguous PFN."""
    buffer = bytearray(0xA000)
    if kind == "x86":
        address = 0xC0000000
        struct.pack_into("<I", buffer, 0x1000 + 768 * 4, 0x2003)
        struct.pack_into("<II", buffer, 0x2000, 0x8003, 0x6003)
    elif kind == "pae":
        address = 0xC0000000
        struct.pack_into("<Q", buffer, 0x1000 + 3 * 8, 0x2003)
        struct.pack_into("<Q", buffer, 0x2000, 0x3003)
        struct.pack_into("<QQ", buffer, 0x3000, 0x8003, 0x6003)
    else:
        address = 0xFFFFFFFF80000000
        struct.pack_into("<Q", buffer, 0x1000 + 511 * 8, 0x2003)
        struct.pack_into("<Q", buffer, 0x2000 + 510 * 8, 0x3003)
        struct.pack_into("<Q", buffer, 0x3000, 0x4003)
        struct.pack_into("<QQ", buffer, 0x4000, 0x8003, 0x6003)
    buffer[0x8000 : 0x8000 + len(BANNER)] = BANNER
    buffer[0x8FFE:0x9000] = b"AB"
    buffer[0x6000:0x6002] = b"CD"
    return memory_context(buffer), address


@pytest.mark.parametrize(
    "kind,layer_class",
    [
        ("x86", intel.LinuxIntel),
        ("pae", intel.LinuxIntelPAE),
        ("x64", intel.LinuxIntel32e),
    ],
)
def test_linux_intel_translation_page_boundary(kind, layer_class):
    context, address = intel_memory(kind)
    layer = layer_class(context, "test", "kernel")
    context.layers.add_layer(layer)
    assert layer.read(address, len(BANNER)) == BANNER
    assert layer.read(address + 0xFFE, 4) == b"ABCD"
    assert layer.translate(address + 0x1000) == (0x6000, "physical")
    with pytest.raises(exceptions.InvalidAddressException):
        layer.read(address + 0x2000, 1)


@pytest.mark.parametrize("userspace", [False, True])
def test_arm64_4k_translation_page_boundary_and_address_space(userspace):
    buffer = bytearray(0xA000)
    # 48-bit VA, four translation levels, 4 KiB granules.
    for table, child in [(0x1000, 0x2000), (0x2000, 0x3000), (0x3000, 0x4000)]:
        struct.pack_into("<Q", buffer, table, child | 3)
    struct.pack_into("<QQ", buffer, 0x4000, 0x8403, 0x6403)
    buffer[0x8FFE:0x9000] = b"AB"
    buffer[0x6000:0x6002] = b"CD"
    context = memory_context(buffer)
    context.config["test.entry_format"] = "<Q"
    context.config["test.TCR_EL1"] = (2 << 30) | (16 << 16) | 16
    # The PR's register-field helper encodes its existing config convention.
    context.config["test.TTBR1_EL1"] = arm.set_reg_bits(
        0x9000 if userspace else 0x1000, arm.AArch64RegMap.TTBR1_EL1.BADDR
    )
    layer = arm.LinuxAArch64(context, "test", "kernel")
    context.layers.add_layer(layer)
    address = 0 if userspace else 0xFFFF000000000000
    assert layer.translate(address + 0x123) == (0x8123, "physical")
    assert layer.read(address + 0xFFE, 4) == b"ABCD"
    with pytest.raises(exceptions.InvalidAddressException):
        layer.read(address + 0x2000, 1)
    with pytest.raises(exceptions.InvalidAddressException):
        layer.read(0xFFFF000000000000 if userspace else 0, 1)


@pytest.mark.parametrize(
    "kind,symbol_name,pkmap_count,layer_class",
    [
        ("x86", "swapper_pg_dir", None, intel.LinuxIntel),
        ("pae", "swapper_pg_dir", 512, intel.LinuxIntelPAE),
        ("pae", "swapper_pg_dir", 2048, intel.LinuxIntelPAE),
        ("x64", "init_level4_pgt", None, intel.LinuxIntel32e),
        ("x64", "init_top_pgt", None, intel.LinuxIntel32e),
    ],
)
@pytest.mark.parametrize("relocated", [False, True])
def test_intel_stacker_selects_layer_and_verifies_actual_banner(
    kind, symbol_name, pkmap_count, layer_class, relocated
):
    context, address = intel_memory(kind)
    symbol_base = address - 0x200000 if relocated else address
    symbols = {
        symbol_name: SimpleNamespace(address=symbol_base + 0x1000),
        "linux_banner": SimpleNamespace(address=symbol_base),
    }
    if pkmap_count is not None:
        symbols["pkmap_count"] = SimpleNamespace(
            type=SimpleNamespace(count=pkmap_count)
        )
    table = SimpleNamespace(symbols=symbols, get_symbol=symbols.__getitem__)
    # PR1088 find_aslr returns physical minus unrelocated virtual address.
    shifts = (-symbol_base, 0x200000) if relocated else (0, 0)
    parent = SimpleNamespace(
        find_aslr=lambda *args, **kwargs: shifts,
        verify_translation_by_banner=LinuxStacker.verify_translation_by_banner,
    )
    layer = LinuxIntelSubStacker(parent).stack(
        context, "physical", table, "symbols", "test", "kernel", BANNER
    )
    assert type(layer) is layer_class
    assert layer.config["page_map_offset"] == 0x1000
    assert layer.config["kernel_virtual_offset"] == shifts[1]
    assert layer.read(address, len(BANNER)) == BANNER


@pytest.mark.parametrize(
    "pgd,pae,expected_arch",
    [
        (0xC0201000, "n", (True, False)),
        (0xC0201000, "y", (True, True)),
        (0xFFFFFFFF80201000, "n", (False, False)),
    ],
)
def test_vmcoreinfo_relocation_and_architecture(pgd, pae, expected_arch):
    note = {
        "NUMBER(phys_base)": 0x400000,
        "KERNELOFFSET": 0x200000,
        "SYMBOL(swapper_pg_dir)": pgd,
        "CONFIG_X86_PAE": pae,
    }
    stacker = LinuxIntelVMCOREINFOStacker
    assert stacker._vmcoreinfo_find_aslr(note) == (0x600000, 0x200000)
    assert stacker._vmcoreinfo_get_dtb(note, 0x200000, 0x600000) == 0x601000
    assert stacker._vmcoreinfo_is_32bit(note) == expected_arch


@pytest.mark.parametrize("missing", ["NUMBER(phys_base)", "KERNELOFFSET"])
def test_vmcoreinfo_missing_relocation_data_is_rejected(missing):
    note = {"NUMBER(phys_base)": 0, "KERNELOFFSET": 0}
    del note[missing]
    assert LinuxIntelVMCOREINFOStacker._vmcoreinfo_find_aslr(note) is None


def test_vmcoreinfo_missing_pgd_is_rejected():
    assert LinuxIntelVMCOREINFOStacker._vmcoreinfo_get_dtb({}, 0, 0) is None
