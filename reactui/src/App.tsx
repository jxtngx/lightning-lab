import './App.css'
import { Container, Grid } from '@mui/material';
import { MetricOne } from "./components/metricOne";
import {MetricTwo} from "./components/metricTwo";
import {MetricThree} from "./components/metricThree";
import {MetricFour} from "./components/metricFour";

const Page = () => (
    <Container maxWidth={false}>
        <Grid
        container
        spacing={3}
        >
            <Grid
            item
            lg={3}
            sm={6}
            xl={3}
            xs={12}
            >
            <MetricOne />
            </Grid>
            <Grid
            item
            xl={3}
            lg={3}
            sm={6}
            xs={12}
            >
            <MetricTwo />
            </Grid>
            <Grid
            item
            xl={3}
            lg={3}
            sm={6}
            xs={12}
            >
            <MetricThree />
            </Grid>
            <Grid
            item
            xl={3}
            lg={3}
            sm={6}
            xs={12}
            >
            <MetricFour />
            </Grid>
        </Grid>
    </Container>
);


export default Page;
