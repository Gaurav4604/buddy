import { createSlice, PayloadAction } from "@reduxjs/toolkit";

export type Reference = {
  file: string;
  chapter: string;
  page: string;
};

type Message = {
  message: string;
  role: "assistant" | "user";
  references?: Reference[];
};

interface ChatState {
  messages: Message[];
}

const initialState: ChatState = {
  messages: [],
};

const chatSlice = createSlice({
  name: "chat",
  initialState,
  reducers: {
    setMessages(state, action: PayloadAction<Message[]>) {
      state.messages = action.payload;
    },
    addMessage(state, action: PayloadAction<Message>) {
      state.messages.push(action.payload);
    },
  },
});

export const { setMessages, addMessage } = chatSlice.actions;
export default chatSlice.reducer;
