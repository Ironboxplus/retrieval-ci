import { Component, type ReactNode } from 'react';
import { TriangleAlert } from 'lucide-react';

import { Button } from '@/components/ui/button';

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * App-level error boundary. Keeps a single page's render crash (e.g. a blocked
 * API returning HTML instead of JSON on the public edge) from white-screening
 * the whole SPA — the nav stays, the user can navigate elsewhere.
 */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: unknown) {
    console.error('[ErrorBoundary]', error, info);
  }

  private reset = () => this.setState({ error: null });

  render() {
    if (this.state.error) {
      return (
        <div className="flex flex-col items-center gap-3 py-20 text-center">
          <TriangleAlert className="size-10 text-warning" />
          <h2 className="text-lg font-semibold">页面加载出错</h2>
          <p className="max-w-lg text-sm text-muted-foreground">{this.state.error.message}</p>
          <Button onClick={this.reset}>重试</Button>
        </div>
      );
    }
    return this.props.children;
  }
}
