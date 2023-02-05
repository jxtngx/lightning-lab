import { Container, Grid } from "@mui/material";

import { MetricOne } from "./metricOne";
import { MetricTwo } from "./metricTwo";
import { MetricThree } from "./metricThree";
import { MetricFour } from "./metricFour";

export const MetricsGrid = () => (
    <Container maxWidth={false}>
        {/* Metric Cards */}
        <Grid container spacing={3}>
            {/* Metric One */}
            <Grid item lg={3} sm={6} xl={3} xs={12}>
                <MetricOne />
            </Grid>
            {/* Metric Two */}
            <Grid item xl={3} lg={3} sm={6} xs={12}>
                <MetricTwo />
            </Grid>
            {/* Metric Three */}
            <Grid item xl={3} lg={3} sm={6} xs={12}>
                <MetricThree />
            </Grid>
            {/* Metric Four */}
            <Grid item xl={3} lg={3} sm={6} xs={12}>
                <MetricFour />
            </Grid>
        </Grid>
    </Container>
);
