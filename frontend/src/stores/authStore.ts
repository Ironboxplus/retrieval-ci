import { create } from 'zustand';

export type AccountTab = 'topup' | 'keys' | 'bindings';

/**
 * Account-modal UI state, lifted to a tiny global store so that any surface can
 * open it — notably the 402 "余额不足" handler in Search/MatchDoc, which lives
 * far from the header's AuthMenu that actually renders the modal.
 */
interface AuthUIState {
  accountOpen: boolean;
  accountTab: AccountTab;
  openAccount: (tab?: AccountTab) => void;
  closeAccount: () => void;
  // Login chooser (3-provider modal), lifted to the store so any page can prompt
  // login on a 401 (paper detail / browse / search) — the modal is mounted in AuthMenu.
  loginOpen: boolean;
  openLogin: () => void;
  closeLogin: () => void;
}

export const useAuthUI = create<AuthUIState>((set) => ({
  accountOpen: false,
  accountTab: 'topup',
  openAccount: (tab = 'topup') => set({ accountOpen: true, accountTab: tab }),
  closeAccount: () => set({ accountOpen: false }),
  loginOpen: false,
  openLogin: () => set({ loginOpen: true }),
  closeLogin: () => set({ loginOpen: false }),
}));
