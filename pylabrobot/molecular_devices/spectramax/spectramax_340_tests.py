import math
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pylabrobot.molecular_devices.spectramax.base import (
  MolecularDevicesPlateReader,
  MolecularDevicesSettings,
)
from pylabrobot.molecular_devices.spectramax.errors import SpectraMaxError
from pylabrobot.molecular_devices.spectramax.spectramax_340 import SpectraMax340
from pylabrobot.resources.agenbio.plates import AGenBio_96_wellplate_Ub_2200ul

# Real !TRANSFER payload captured from a SpectraMax 340 (450 nm endpoint, water), with a negative
# and a saturated value substituted to cover those cases.
TRANSFER_340 = (
  "6510.3\t 21.1\r\nL:\t 450\r\nL:\t450\t\r\n"
  " 1:\t    .028\t    .030\t    .029\t    .032\t    .029\t    .029\t    .029\t    .028\t\r\n"
  " 2:\t    .029\t    .029\t    .047\t    .030\t    .029\t    .031\t    .030\t    .029\t\r\n"
  " 3:\t    .030\t    .030\t    .030\t    .029\t    .027\t    .031\t    .030\t    .031\t\r\n"
  " 4:\t    .031\t    .030\t    .026\t    .030\t    .030\t    .030\t    .030\t    .031\t\r\n"
  " 5:\t    .030\t    .030\t    .029\t    .029\t    .029\t    .030\t    .030\t    .029\t\r\n"
  " 6:\t    .030\t    .029\t    .030\t    .033\t    .031\t    .032\t    .031\t    .033\t\r\n"
  " 7:\t    .031\t    .030\t    .031\t    .029\t    .030\t    .032\t    .030\t    .032\t\r\n"
  " 8:\t    .031\t    .030\t    .031\t    .034\t    .033\t    .031\t    .032\t    .030\t\r\n"
  " 9:\t    .032\t    .030\t    .029\t    .033\t    .031\t    .031\t    .034\t    .033\t\r\n"
  "10:\t    .031\t    .032\t    .033\t    .032\t    .032\t    .033\t    .031\t    .029\t\r\n"
  "11:\t    .031\t    .032\t    .030\t    .034\t    .030\t    .030\t    .031\t   -.005\t\r\n"
  "12:\t    .030\t    .044\t    .032\t    .031\t    .030\t    .030\t    .029\t    #SAT"
)


class TestParseValue(unittest.TestCase):
  def test_formats(self):
    p = MolecularDevicesPlateReader._parse_value
    self.assertEqual(p(".028"), 0.028)
    self.assertEqual(p("-.005"), -0.005)
    self.assertEqual(p("1.234"), 1.234)
    self.assertEqual(p("-1.5"), -1.5)
    self.assertEqual(p("2"), 2.0)
    self.assertEqual(p("1.2E-3"), 0.0012)
    self.assertTrue(math.isinf(p("#SAT")))
    self.assertTrue(math.isnan(p("#ERR")))
    self.assertTrue(math.isnan(p("")))


class TestSpectraMax340(unittest.IsolatedAsyncioTestCase):
  def setUp(self):
    self.mock_serial = MagicMock()
    self.mock_serial.setup = AsyncMock()
    self.mock_serial.stop = AsyncMock()
    self.mock_serial.write = AsyncMock()
    with patch("pylabrobot.io.serial.Serial", return_value=self.mock_serial):
      self.device = SpectraMax340(name="sm340", port="COM1")
      self.device.io = self.mock_serial

  def _settings(self, **kw):
    plate = AGenBio_96_wellplate_Ub_2200ul("plate")
    base = dict(
      plate=plate,
      read_mode="absorbance",
      read_type="endpoint",
      read_order="column",
      calibrate="once",
      shake_settings=None,
      carriage_speed="normal",
      speed_read=False,
      kinetic_settings=None,
      spectrum_settings=None,
      wavelengths=[450],
    )
    base.update(kw)
    return MolecularDevicesSettings(**base)

  def test_parse_real_transfer(self):
    (m,) = self.device._parse_data(TRANSFER_340, self._settings())
    self.assertEqual(m["wavelength"], 450)
    self.assertEqual(m["temperature"], 21.1)
    data = m["data"]
    self.assertEqual((len(data), len(data[0])), (8, 12))
    self.assertEqual(data[0][0], 0.028)  # A1
    self.assertEqual(data[2][1], 0.047)  # C2
    self.assertEqual(data[7][10], -0.005)  # H11: negative values are kept
    self.assertTrue(math.isinf(data[7][11]))  # H12: #SAT

  async def test_read_absorbance_skips_unsupported_commands(self):
    send = patch.object(self.device, "send_command", new_callable=AsyncMock).start()
    self.addCleanup(patch.stopall)
    with (
      patch.object(self.device, "_read_now", new_callable=AsyncMock),
      patch.object(self.device, "wait_for_idle", new_callable=AsyncMock),
      patch.object(
        self.device,
        "_transfer_data",
        new_callable=AsyncMock,
        return_value=[{"data": [[0.1]], "wavelength": 450, "temperature": 21.0, "time": 1.0}],
      ),
    ):
      plate = AGenBio_96_wellplate_Ub_2200ul("plate")
      await self.device.read_absorbance(plate, plate.get_wells(), 450)
    commands = [c.args[0] for c in send.call_args_list]
    for unsupported in ("!XPOS", "!YPOS", "!CSPEED", "!TAG", "!READTYPE", "!NVRAM"):
      self.assertFalse(any(c.startswith(unsupported) for c in commands), unsupported)
    for expected in (
      "!CLEAR DATA",
      "!STRIP 1 12",
      "!SHAKE OFF",
      "!WAVELENGTH 450",
      "!CALIBRATE ONCE",
      "!MODE ENDPOINT",
      "!ORDER COLUMN",
    ):
      self.assertIn(expected, commands)
    speed = next(c for c in send.call_args_list if c.args[0].startswith("!SPEED"))
    self.assertEqual((speed.args[0], speed.kwargs), ("!SPEED OFF", {"num_res_fields": 1}))

  async def test_readtype_rejects_non_absorbance(self):
    with self.assertRaises(NotImplementedError):
      await self.device._set_readtype(self._settings(read_mode="fluorescence"))

  async def test_error_log_empty_single_field(self):
    chunks = iter([b"OK\t\r\n", b">"])
    # After the reply, readline keeps returning b"" (serial timeout) during the grace period.
    self.mock_serial.readline = AsyncMock(side_effect=lambda: next(chunks, b""))
    self.assertEqual(await self.device.read_error_log(), [])

  def _serial_replies(self, *chunks):
    it = iter(chunks)
    self.mock_serial.readline = AsyncMock(side_effect=lambda: next(it, b""))

  async def test_transfer_next_reading_not_ready(self):
    self._serial_replies(b"FAIL\t 411\t\r\n", b">")
    self.assertIsNone(await self.device.transfer_next_reading(self._settings()))

  async def test_transfer_next_reading_data(self):
    self._serial_replies(b"OK\t\r\n", b">", TRANSFER_340.encode(), b"\r\n>")
    (m,) = await self.device.transfer_next_reading(self._settings())
    self.assertEqual(m["data"][0][0], 0.028)

  async def test_transfer_other_error_raises(self):
    self._serial_replies(b"FAIL\t 100\t\r\n", b">")
    with self.assertRaises(SpectraMaxError) as ctx:
      await self.device.transfer_next_reading(self._settings())
    self.assertEqual(ctx.exception.code, 100)

  async def test_error_103_is_too_many_arguments(self):
    with self.assertRaisesRegex(SpectraMaxError, "error 103: too many arguments"):
      self.device._parse_basic_errors(["FAIL\t 103"], "!WAVELENGTH 1 2 3 4 5 6 7")

  async def test_rejects_unsupported_settings_before_sending(self):
    send = patch.object(self.device, "send_command", new_callable=AsyncMock).start()
    self.addCleanup(patch.stopall)
    with self.assertRaises(ValueError):
      await self.device._set_wavelengths(
        self._settings(wavelengths=[400, 410, 420, 430, 440, 450, 460])
      )
    with self.assertRaises(NotImplementedError):
      await self.device._set_mode(self._settings(read_type="spectrum"))
    with self.assertRaises(NotImplementedError):
      await self.device._set_wavelengths(self._settings(path_check=True))
    send.assert_not_called()

  async def test_open_waits_for_status(self):
    send = patch.object(self.device, "send_command", new_callable=AsyncMock).start()
    self.addCleanup(patch.stopall)
    with patch.object(
      self.device,
      "request_status",
      new_callable=AsyncMock,
      side_effect=[["CLOSED", "IDLE"], ["OPEN", "IDLE"]],
    ) as status:
      await self.device.open()
    send.assert_called_once_with("!OPEN")
    self.assertEqual(status.call_count, 2)

  async def test_set_temperature_single_field(self):
    send = patch.object(self.device, "send_command", new_callable=AsyncMock).start()
    self.addCleanup(patch.stopall)
    await self.device.set_temperature(37)
    await self.device.deactivate()
    self.assertEqual([c.args for c in send.call_args_list], [("!TEMP 37",), ("!TEMP 0",)])
    self.assertTrue(all(c.kwargs == {"num_res_fields": 1} for c in send.call_args_list))

  async def test_error_log_with_entries(self):
    self.mock_serial.readline = AsyncMock(
      side_effect=[b"OK\t\r\n", b">", b"100\tcommand not found (3 times)\r\n", b">"]
    )
    self.assertEqual(
      await self.device.read_error_log(), ["100", "command", "not", "found", "(3", "times)"]
    )
