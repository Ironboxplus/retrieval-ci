import { toast } from 'sonner';

import ConfirmButton from '../ConfirmButton';
import RunView from './RunView';
import { useCancelTask } from '../../api/hooks/useTasks';
import { useTaskStream } from '../../api/hooks/useTaskStream';
import { MAX_STREAM_EVENTS } from '../../api/hooks/taskStreamModel';
import { apiErrorText, useTaskReplay } from '../../api/hooks/useSchedules';

/** A task that is still running: stage chain and events arrive over SSE. */
export function LiveRun({ taskId, startedAtMs }: { taskId: number; startedAtMs?: number | null }) {
  const stream = useTaskStream(taskId);
  const cancel = useCancelTask();
  const active = stream.task != null && (stream.task.status === 'running' || stream.task.status === 'pending');
  return (
    <RunView
      task={stream.task}
      nodes={stream.nodes}
      events={stream.events}
      dropped={stream.dropped}
      startedAtMs={startedAtMs}
      connection={stream.connection}
      error={stream.error}
      resource={stream.resource}
      actions={active && (
        <ConfirmButton
          variant="destructive" title="取消该任务？" confirmLabel="取消任务" loading={cancel.isPending}
          onConfirm={() => cancel.mutate(taskId, { onError: (e) => toast.error(apiErrorText(e)) })}
        >
          取消
        </ConfirmButton>
      )}
    />
  );
}

/** A finished task read back from storage. */
export function ReplayRun({ taskId }: { taskId: number }) {
  const replay = useTaskReplay(taskId, MAX_STREAM_EVENTS);
  return (
    <RunView
      task={replay.data?.task ?? null}
      nodes={replay.data?.nodes ?? []}
      events={replay.data?.events ?? []}
      error={replay.isError ? apiErrorText(replay.error) : null}
    />
  );
}
