<template>
  <button
    type="button"
    class="inline-flex min-h-0 items-center rounded-box transition-colors disabled:pointer-events-none"
    :class="[large ? 'px-2.5 h-9 gap-1.5 text-sm' : 'px-2 h-8 gap-1 text-xs', buttonClass]"
    :disabled="disabled"
    @click="emit('click', $event)"
  >
    <component
      v-if="icon"
      :is="icon"
      class="shrink-0"
      :class="large ? 'h-4 w-4' : 'h-3.5 w-3.5'"
    />
    <span class="whitespace-nowrap">
      <slot />
    </span>
  </button>
</template>

<script setup lang="ts">
import { computed, type Component } from 'vue';

const props = withDefaults(defineProps<{
  icon?: Component;
  disabled?: boolean;
  danger?: boolean;
  primary?: boolean;
  selected?: boolean;
  large?: boolean;
}>(), {
  disabled: false,
  danger: false,
  primary: false,
  selected: false,
  large: false,
});

const emit = defineEmits<{
  click: [event: MouseEvent];
}>();

const buttonClass = computed(() => {
  if (props.disabled) return 'text-base-content/30';
  if (props.danger) return 'text-error/70 hover:text-error hover:bg-error/10 cursor-pointer';
  if (props.primary) return 'text-primary-content/70 bg-primary/70 hover:text-primary-content hover:bg-primary cursor-pointer';
  if (props.selected) return 'text-base-content/70 bg-base-100/30 hover:text-base-content cursor-pointer';
  return 'text-base-content/70 hover:bg-base-100 hover:text-base-content cursor-pointer';
});
</script>
