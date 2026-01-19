"""
Bank-specific views package.

This package contains bank-specific views that handle sensitive matching algorithms
and bank-specific processing logic. Each bank has its own module:
- bt.py: BT (Banque de Tunisie) specific views
- stb.py: STB (Société Tunisienne de Banque) specific views
- attijari.py: ATTIJARI BANK specific views

All bank views inherit from base classes defined in base.py
"""

from . import base, bt, stb, attijari

__all__ = ['base', 'bt', 'stb', 'attijari']



