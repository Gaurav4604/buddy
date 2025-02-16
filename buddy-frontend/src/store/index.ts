import { configureStore } from "@reduxjs/toolkit";
import chatReducer from "./chatSlice";
import topicsAndContentReducer from "./topicAndContentSlice";

const store = configureStore({
  reducer: {
    chat: chatReducer,
    tnc: topicsAndContentReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

export default store;
