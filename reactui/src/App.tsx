import "./App.css";
import Container from "@mui/material/Container";
import NavBar from "./components/mainPage/navBar";
import { MetricsGrid } from "./components/metrics/metricsGrid";

const Page = () => (
    <>
    <NavBar />
    <br/><br/>
    <Container>
         <MetricsGrid />
    </Container>
    </>
);

export default Page;
