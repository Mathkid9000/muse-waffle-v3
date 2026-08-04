import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import Editor from "./pages/Editor";
import ProtectedRoute from "./pages/ProtectedRoute";

export default function App() {
    return (
        <AuthProvider>
            <BrowserRouter basename={import.meta.env.BASE_URL}>
                <Routes>
                    <Route path="/login" element={ <Login/> } />
                    <Route path="/" element={ <Navigate to="/dashboard" replace /> } />
                    <Route path="/dashboard" element={
                        <ProtectedRoute><Dashboard /></ProtectedRoute>
                    } />
                    <Route path="/editor" element={
                        <ProtectedRoute><Editor /></ProtectedRoute>
                    } />
                    <Route path="/editor/:id" element={
                        <ProtectedRoute><Editor /></ProtectedRoute>
                    } />
                </Routes>
            </BrowserRouter>
        </AuthProvider>
    )
}