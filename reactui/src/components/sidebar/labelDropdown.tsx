import "../../App.css";
import { Card, CardContent, Grid, Typography } from "@mui/material";

export const LabelDropdown = (props: any) => (
    <Card {...props}>
        <CardContent>
            <Grid container spacing={3}>
                <Grid item>
                    <Typography variant="h5">Label Dropdown</Typography>
                </Grid>
            </Grid>
        </CardContent>
    </Card>
);
