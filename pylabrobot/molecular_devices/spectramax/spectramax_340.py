import asyncio
import time
from typing import Dict, List, Optional

from pylabrobot.resources import Coordinate, PlateHolder

from .base import RES_TERM_CHAR, MolecularDevicesPlateReader, MolecularDevicesSettings
from .errors import ERROR_CODES, SpectraMaxError, SpectraMaxUnrecognizedCommandError

NO_DATA_AVAILABLE = 411  # !TRANSFER before the next reading exists


class SpectraMax340(MolecularDevicesPlateReader):
  """Molecular Devices SpectraMax 340 plate reader (96-well, absorbance only, 340-750 nm).

  The 340 firmware (``!OPTION`` reports ``SPECTRAmax 340 00A``) speaks an older dialect of the
  Molecular Devices protocol. Verified on the instrument:

  - Not implemented (error 100 "command not found"): ``!XPOS``/``!YPOS`` (fixed plate geometry),
    ``!CSPEED``, ``!TAG``, ``!READTYPE``. The corresponding setup steps are skipped.
  - ``!NVRAM CARCOL 100`` does not set the carriage settle time: the firmware answers with a hex
    memory dump and an extra ``>`` prompt, which desynchronises every following reply. Never sent.
  - ``!SPEED``, ``!TEMP <value>`` and ``!ERROR`` (with an empty error log) answer with one field.
  - No spectrum mode (``!MODE SPECTRUM`` rejects every argument combination; no spectrum on the
    front panel). At most 6 wavelengths per read.
  - Native kinetic works, and ``!TRANSFER`` returns each reading as soon as it is measured
    (error 411 "no data available" before that): see ``transfer_next_reading``.
  - ``!OPEN``/``!CLOSE`` move the drawer; the status reports OPEN/CLOSED after ~1-2 s.
  """

  # The 340's own error log reports 103 as "too many arguments" (verified: !WAVELENGTH with more
  # than 6 values), unlike the table of newer models.
  error_codes = {
    **ERROR_CODES,
    103: ("too many arguments", SpectraMaxUnrecognizedCommandError),
  }
  MAX_WAVELENGTHS = 6

  def __init__(self, name: str, port: str):
    super().__init__(port=port, human_readable_device_name="Molecular Devices SpectraMax 340")
    self.loading_tray = PlateHolder(
      name=name + "_loading_tray",
      size_x=127.76,
      size_y=85.48,
      size_z=0,  # TODO: measure
      pedestal_size_z=0,  # TODO: measure
      child_location=Coordinate.zero(),  # TODO: measure
    )

  async def request_firmware_version(self) -> List[str]:
    """Get the firmware version.

    ``COMMAND_TERMINATORS`` lists ``!OPTION`` under the key ``"OPTION"``, so the lookup in
    ``send_command`` misses it and reads only the first of the two response fields. Pass the field
    count explicitly so the payload is not left in the serial buffer.
    """
    res = await self.send_command("!OPTION", num_res_fields=2)
    return res[1].split()

  # -- drawer ---------------------------------------------------------------------------------

  async def open(self, timeout: float = 15) -> None:
    """Open the drawer and wait until the status reports OPEN (~1-2 s)."""
    await self.send_command("!OPEN")
    await self._wait_drawer("OPEN", timeout)

  async def close(self, timeout: float = 15) -> None:
    """Close the drawer and wait until the status reports CLOSED (~1-2 s)."""
    await self.send_command("!CLOSE")
    await self._wait_drawer("CLOSED", timeout)

  async def _wait_drawer(self, state: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while True:
      status = await self.request_status()
      if status and status[0] == state:
        return
      if time.time() > deadline:
        raise TimeoutError(f"Drawer did not report {state} within {timeout} s (status {status}).")
      await asyncio.sleep(0.3)

  # -- reads ----------------------------------------------------------------------------------

  async def start_read(self, settings: MolecularDevicesSettings) -> None:
    """Configure and start a read without waiting for it to finish (see transfer_next_reading).

    Same setup sequence as ``read_absorbance``; use it for native kinetic runs whose readings
    should be retrieved one by one while the reader keeps the timing.
    """
    await self._set_clear()
    await self._set_strip(settings)
    await self._set_shake(settings)
    await self._set_wavelengths(settings)
    await self._set_calibrate(settings)
    await self._set_mode(settings)
    await self._set_order(settings)
    await self._set_speed(settings)
    await self._read_now()

  async def transfer_next_reading(
    self, settings: MolecularDevicesSettings, timeout: float = 30
  ) -> Optional[List[Dict]]:
    """Fetch the next reading of the current read with ``!TRANSFER``.

    Returns the parsed measurements (one dict per wavelength), or ``None`` if the next reading is
    not available yet (firmware error 411). During a native kinetic read, each reading becomes
    available as soon as it has been measured, so this can be polled for live data.
    """
    await self.io.write(b"!TRANSFER\r")
    raw = b""
    deadline = time.time() + timeout
    # First field: "OK" (data follows in a second field) or "FAIL <code>" (no second field).
    while raw.count(RES_TERM_CHAR) < 1:
      if time.time() > deadline:
        raise TimeoutError("Timeout waiting for response to command: !TRANSFER")
      raw += await self.io.readline()
    if b"FAIL" not in raw.split(RES_TERM_CHAR)[0]:
      while raw.count(RES_TERM_CHAR) < 2:
        if time.time() > deadline:
          raise TimeoutError("Timeout waiting for !TRANSFER data")
        raw += await self.io.readline()
    response = [r.strip() for r in raw.decode("utf-8", errors="replace").split(">") if r.strip()]
    try:
      self._parse_basic_errors(response, "!TRANSFER")
    except SpectraMaxError as e:
      if getattr(e, "code", None) == NO_DATA_AVAILABLE:
        return None
      raise
    return self._parse_data(response[1], settings)

  async def _set_mode(self, settings: MolecularDevicesSettings) -> None:
    if settings.read_type not in ("endpoint", "kinetic"):
      raise NotImplementedError(f"The SpectraMax 340 has no {settings.read_type} mode.")
    await super()._set_mode(settings)

  async def _set_wavelengths(self, settings: MolecularDevicesSettings) -> None:
    # More than 6 values makes the firmware answer once per extra value, desyncing replies.
    if len(settings.wavelengths) > self.MAX_WAVELENGTHS:
      raise ValueError(f"The SpectraMax 340 reads at most {self.MAX_WAVELENGTHS} wavelengths.")
    if settings.path_check:
      raise NotImplementedError("PathCheck is not available on the SpectraMax 340.")
    await super()._set_wavelengths(settings)

  # -- temperature ----------------------------------------------------------------------------

  async def set_temperature(self, temperature: float) -> None:
    """Set the incubator setpoint (0 = off). The 340 answers ``!TEMP <value>`` with one field."""
    if not (0 <= temperature <= 45):
      raise ValueError("Temperature must be between 0 and 45°C.")
    await self.send_command(f"!TEMP {temperature}", num_res_fields=1)

  async def deactivate(self) -> None:
    await self.set_temperature(0)

  async def read_error_log(self) -> List[str]:
    """Read the error log. Empty log: the 340 answers ``OK>`` only (one field)."""
    res = await self._send_command_variable_fields("!ERROR")
    return res[1].split() if len(res) > 1 else []

  async def _send_command_variable_fields(
    self, command: str, timeout: float = 10, grace: float = 0.5
  ) -> List[str]:
    """Send a command whose reply has one or two ``>``-terminated fields.

    Reads until the first ``>``, then keeps reading for ``grace`` seconds in case a second field
    follows, so nothing is left in the serial buffer for the next command.
    """
    await self.io.write(command.encode() + b"\r")
    raw = b""
    deadline = time.time() + timeout
    while raw.count(RES_TERM_CHAR) < 1:
      if time.time() > deadline:
        raise TimeoutError(f"Timeout waiting for response to command: {command}")
      raw += await self.io.readline()
    grace_end = time.time() + grace
    while raw.count(RES_TERM_CHAR) < 2 and time.time() < grace_end:
      raw += await self.io.readline()
    response = raw.decode("utf-8", errors="replace").strip().split(RES_TERM_CHAR.decode())
    response = [r.strip() for r in response if r.strip() != ""]
    self._parse_basic_errors(response, command)
    return response

  # -- setup steps not supported by the 340 firmware --------------------------------------------

  async def _set_plate_position(self, settings: MolecularDevicesSettings) -> None:
    """No ``!XPOS``/``!YPOS``: the 340 uses a fixed 96-well geometry."""

  async def _set_carriage_speed(self, settings: MolecularDevicesSettings) -> None:
    """No ``!CSPEED`` on the 340."""

  async def _set_tag(self, settings: MolecularDevicesSettings) -> None:
    """No ``!TAG`` on the 340."""

  async def _set_readtype(self, settings: MolecularDevicesSettings) -> None:
    """No ``!READTYPE``: the 340 is an absorbance plate reader only."""
    if settings.read_mode != "absorbance" or settings.cuvette:
      raise NotImplementedError("The SpectraMax 340 only reads absorbance in plates.")

  async def _set_nvram(self, settings: MolecularDevicesSettings) -> None:
    """Never send ``!NVRAM``: on the 340 it dumps memory and desynchronises replies."""

  async def _set_speed(self, settings: MolecularDevicesSettings) -> None:
    await self.send_command(f"!SPEED {'ON' if settings.speed_read else 'OFF'}", num_res_fields=1)
