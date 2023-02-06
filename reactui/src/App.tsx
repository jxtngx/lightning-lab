import "./App.css";
import { Container, Grid } from "@mui/material";
import NavBar from "./components/mainPage/navBar";
import { MetricsGrid } from "./components/metrics/metricsGrid";
import { SideBarGrid } from "./components/sidebar/sidebarGrid";
import { ImageGrid } from "./components/graphs/imageGrid";

const Page = () => (
    <>
        <NavBar />
        <br />
        <br />
        <Container>
            <Grid container>
                <Grid>
                    <SideBarGrid />
                </Grid>
                <Grid>
                    <MetricsGrid />
                    <br />
                    <br />
                    <ImageGrid />
                </Grid>
            </Grid>
        </Container>
    </>
);

export default Page;
