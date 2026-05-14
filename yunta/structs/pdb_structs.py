"""Data structures for PDB data."""

from dataclasses import dataclass, field

@dataclass
class ATMRecord:

    """Parse an ATM record from a PDB line.

    """

    line: str
    name: str = field(init=False)
    atm_no: int = field(init=False)
    atm_name: str = field(init=False)
    atm_alt: str = field(init=False)
    res_name: str = field(init=False)
    chain: str = field(init=False)
    res_no: int = field(init=False)
    insert: str = field(init=False)
    resid: str = field(init=False)
    x: float = field(init=False)
    y: float = field(init=False)
    z: float = field(init=False)
    occ: float = field(init=False)
    B: float = field(init=False)

    def __post_init__(self):
        self.name = self.line[0:6].strip()
        self.atm_no = int(self.line[6:11])
        self.atm_name = self.line[12:16].strip()
        self.atm_alt = self.line[17]
        self.res_name = self.line[17:20].strip()
        self.chain = self.line[21]
        self.res_no = int(self.line[22:26])
        self.insert = self.line[26].strip()
        self.resid = self.line[22:29]
        self.x = float(self.line[30:38])
        self.y = float(self.line[38:46])
        self.z = float(self.line[46:54])
        self.occ = float(self.line[54:60])
        self.B = float(self.line[60:66])
