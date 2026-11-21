import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface AppState {
  siderCollapsed: boolean;
  darkMode: boolean;
  toggleSider: () => void;
  toggleDarkMode: () => void;
  setDarkMode: (v: boolean) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      siderCollapsed: false,
      darkMode: false,
      toggleSider: () => set((s) => ({ siderCollapsed: !s.siderCollapsed })),
      toggleDarkMode: () => set((s) => ({ darkMode: !s.darkMode })),
      setDarkMode: (v: boolean) => set({ darkMode: v }),
    }),
    { name: 'retrival-app-store' },
  ),
);
