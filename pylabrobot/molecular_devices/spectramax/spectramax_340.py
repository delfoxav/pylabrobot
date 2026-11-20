from typing import List

from pylabrobot.resources import Coordinate, PlateHolder

from .base import MolecularDevicesPlateReader


class SpectraMax340(MolecularDevicesPlateReader):
  """Molecular Devices SpectraMax 340 plate reader."""

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
